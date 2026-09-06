"""Generate and independently score private fine-tuned/baseline voice samples."""

import argparse
import asyncio
import gc
import importlib.metadata
import json
import math
import sqlite3
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .corpus import atomic_json, sha256
from .data import group_key, read_jsonl, select_reference
from .verify_corpus import keys

NOVEL = {
    "validation": [
        "La estación estaba vacía cuando llegó el último tren. Nadie bajó del vagón, pero en uno de los asientos había una carta con mi nombre.",
        "Habíamos confundido la comodidad con la felicidad. Bastó una avería, una tarde sin pantallas, para recordar todo lo que habíamos dejado de mirar.",
        "¿De verdad esperabas una respuesta sencilla? La historia de aquel pueblo tenía demasiadas puertas cerradas y muy pocas ventanas abiertas.",
        (
            "Al otro lado del puente empezaba el bosque. Caminamos despacio, escuchando la lluvia sobre las hojas, "
            "hasta que la ciudad dejó de oírse. Mi padre conocía aquel camino desde niño, pero esa tarde se detuvo "
            "varias veces, como si los árboles hubieran cambiado de sitio. Cerca de una curva encontramos una casa "
            "de piedra. La puerta estaba abierta y había luz en la cocina. Una mujer salió a recibirnos con dos "
            "tazas de café. No preguntó quiénes éramos ni de dónde veníamos. Se limitó a señalar las sillas junto "
            "a la ventana. Durante unos minutos hablamos del tiempo y del estado del puente. Después mi padre "
            "sacó del bolsillo un sobre amarillo y lo dejó sobre la mesa. La mujer lo miró sin tocarlo. Fuera "
            "seguía lloviendo, y por primera vez comprendí que aquel paseo tenía una razón que nadie me había "
            "contado. No pregunté nada. Me quedé escuchando el agua que caía desde el tejado, mientras los dos "
            "adultos buscaban la forma de empezar una conversación que habían aplazado durante muchos años."
        ),
    ],
    "test": [
        "El reloj de la cocina llevaba meses parado. Aquella mañana, sin que nadie lo tocara, dio las siete. Mi hermana y yo dejamos las tazas sobre la mesa.",
        "Una sociedad puede acostumbrarse a casi todo, incluso al ruido de sus propias contradicciones. Lo difícil es reconocerlas cuando llegan disfrazadas de sentido común.",
        "No era una despedida triste. Habíamos aprendido a compartir el silencio, y ahora también nos tocaba aprender a compartir la distancia.",
        (
            "El librero abrió una caja pequeña y sonrió. Dentro no había joyas ni documentos importantes: "
            "solamente un mapa, una llave y la fotografía de una playa desconocida. Me explicó que la caja "
            "había llegado con los libros de una profesora jubilada. Nadie de su familia sabía qué hacer con "
            "ella, así que terminó olvidada entre dos atlas antiguos. Estudié el mapa con cuidado. No tenía "
            "nombres de ciudades, pero reconocí la forma de una montaña que había visto desde la carretera. "
            "El librero me dejó llevar la fotografía. Al salir, la plaza estaba llena de gente que esperaba "
            "el comienzo de un concierto. Crucé entre las sillas vacías y me senté en el borde de la fuente. "
            "Por detrás de la imagen había una fecha y una frase escrita a lápiz. La leí varias veces antes "
            "de guardar la foto en mi cuaderno. Todavía no sabía adónde me llevaría aquella pista, pero por "
            "primera vez en mucho tiempo tenía ganas de hacer un viaje sin reservar de antemano el billete "
            "de vuelta. Cuando comenzó la música, decidí que iría a buscar la montaña al día siguiente."
        ),
    ],
}


def word_error(expected: str, actual: str) -> tuple[int, int]:
    """Return normalized word edit distance and reference word count."""
    reference, heard = keys(expected), keys(actual)
    previous = list(range(len(heard) + 1))
    for i, word in enumerate(reference, 1):
        current = [i]
        for j, other in enumerate(heard, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (word != other)))
        previous = current
    return previous[-1], len(reference)


def has_complete_ending(expected: str, actual: str) -> bool:
    """Catch premature EOS that a low aggregate word-error rate can hide."""
    reference, heard = keys(expected), keys(actual)
    ending_length = min(3, len(reference))
    return bool(reference) and heard[-ending_length:] == reference[-ending_length:]


def inference_source_identity() -> dict[str, str]:
    """Bind measured audio to the actual local inference and chunking code."""
    root = Path(__file__).resolve().parents[2]
    names = (
        "backend/backends/qwen_finetuned_backend.py",
        "backend/backends/mlx_backend.py",
        "backend/backends/mlx_qwen_optimizations.py",
        "backend/backends/base.py",
        "backend/utils/chunked_tts.py",
        "backend/utils/audio.py",
    )
    return {name: sha256(root / name) for name in names}


def evaluation_cases(corpus: Path, split: str, novel: list[str] | None = None) -> list[dict]:
    """Choose fixed chapter-balanced passages and independently written prose."""
    rows = read_jsonl(corpus / f"{split}_verified.jsonl")
    selected = []
    for source, chapter in sorted({group_key(row) for row in rows}):
        group = sorted((row for row in rows if group_key(row) == (source, chapter)), key=lambda row: row["duration"])
        indices = np.linspace(0, len(group) - 1, min(6, len(group)), dtype=int)
        selected.extend(
            {
                "id": group[i]["id"],
                "text": group[i]["text"],
                "chapter": chapter,
                **({"source": source} if source else {}),
            }
            for i in indices
        )
    passages = NOVEL[split] if novel is None else novel
    if (
        not isinstance(passages, list)
        or len(passages) != 4
        or any(not isinstance(text, str) or not keys(text) for text in passages)
        or len(set(passages)) != 4
        or max(len(keys(text)) for text in passages) < 150
    ):
        raise ValueError(
            "Evaluation requires four distinct new prose passages, including at least 150 words of long form"
        )
    selected.extend({"id": f"novel_{i}", "text": text, "chapter": None} for i, text in enumerate(passages))
    return [{**row, "seed": 781 + i * 17} for i, row in enumerate(selected)]


def local_baseline_voice(data_dir: Path, profile_id: str) -> dict | None:
    """Resolve the actual installed baseline, including a previously fine-tuned voice."""
    from backend.services.finetuned_voices import read_voice

    with sqlite3.connect(f"file:{data_dir / 'voicebox.db'}?mode=ro", uri=True) as db:
        row = db.execute(
            "SELECT voice_type,preset_engine,preset_voice_id FROM profiles WHERE id=?", (profile_id,)
        ).fetchone()
    if row is None:
        raise ValueError("The selected baseline profile does not exist")
    if row[0] == "preset" and row[1] == "qwen_custom_voice" and str(row[2]).startswith("finetuned:"):
        return read_voice(row[2], verify_weights=True)
    if row[0] == "preset":
        raise ValueError("The evaluation baseline must be a cloned or locally fine-tuned Qwen voice")
    return None


def baseline_reference(data_dir: Path, profile_id: str, output: Path) -> tuple[Path, str]:
    """Reconstruct the existing profile's ordered reference without changing it."""
    from backend.backends.base import combine_voice_prompts
    from backend.utils.audio import save_audio

    with sqlite3.connect(f"file:{data_dir / 'voicebox.db'}?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT audio_path,reference_text FROM profile_samples WHERE profile_id=? ORDER BY ordinal,id",
            (profile_id,),
        ).fetchall()
    if not rows:
        raise ValueError("The baseline voice has no saved samples")
    audio, text = asyncio.run(
        combine_voice_prompts([str(data_dir / relative) for relative, _text in rows], [text for _path, text in rows])
    )
    path = output / "baseline-reference.wav"
    save_audio(audio, str(path), 24000)
    return path, text


def synthesize_sample(model, variant, text, seed, speaker, conditioning, reference_text):
    """Run the same bounded warm inference settings used by the two backends."""
    import mlx.core as mx

    from backend.backends.mlx_qwen_optimizations import clear_qwen_reference_decode_handoff

    np.random.seed(seed)
    limit = min(4096, max(75, len(model.tokenizer.encode(text)) * 6))
    before = time.monotonic()
    if variant == "candidate":
        from backend.backends.qwen_finetuned_backend import LocalQwenCustomVoiceBackend
        from backend.utils.chunked_tts import generate_chunked, release_disk_backed_audio

        backend = LocalQwenCustomVoiceBackend()
        backend.model = model
        backend.voice = {"voice_id": "finetuned:evaluation", "speaker": speaker}
        audio = None
        try:
            audio, rate = asyncio.run(
                generate_chunked(
                    backend,
                    text,
                    {"preset_voice_id": backend.voice["voice_id"]},
                    language="es",
                    seed=seed,
                )
            )
            if rate != 24000:
                raise RuntimeError("Evaluation returned an unexpected sample rate")
            result = np.array(audio, dtype=np.float32, copy=True)
        finally:
            release_disk_backed_audio(audio)
        # The real backend rejects a duration-limit hit in any logical chunk.
        # Its public API returns audio, not an aggregate codec-token count.
        return result, time.monotonic() - before, None
    mx.random.seed(seed)
    try:
        generated = list(
            model.generate(
                text, ref_audio=conditioning.audio, ref_text=reference_text, lang_code="spanish", max_tokens=limit
            )
        )
    finally:
        clear_qwen_reference_decode_handoff(model)
    if not generated or any(row.sample_rate != 24000 for row in generated):
        raise RuntimeError("Evaluation returned no audio or an unexpected sample rate")
    audio = np.concatenate([np.asarray(row.audio, dtype=np.float32).reshape(-1) for row in generated])
    elapsed = time.monotonic() - before
    if not len(audio) or not np.isfinite(audio).all():
        raise RuntimeError("Evaluation returned empty or nonfinite audio")
    return audio, elapsed, sum(row.token_count for row in generated)


def speaker_embedding(model, path: Path) -> np.ndarray:
    """Use the unchanged base speaker encoder as an identity comparison."""
    import mlx.core as mx

    audio, rate = sf.read(path, dtype="float32")
    if rate != 24000:
        raise ValueError("Invalid speaker comparison sample rate")
    vector = model.extract_speaker_embedding(mx.array(audio)).astype(mx.float32)
    mx.eval(vector)
    array = np.asarray(vector).reshape(-1)
    if not np.isfinite(array).all() or np.linalg.norm(array) == 0:
        raise RuntimeError("Invalid speaker comparison embedding")
    return array / np.linalg.norm(array)


def generate(config_path: Path) -> None:
    """Generate both models with identical text/seeds and score speaker identity."""
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model

    from backend import config as backend_config
    from backend.backends.mlx_backend import _apply_mlx_audio_qwen_dtype_backport
    from backend.backends.mlx_qwen_optimizations import (
        apply_qwen_icl_cache_backport,
        apply_qwen_reference_decode_backport,
        prepare_reference_conditioning,
    )
    from backend.backends.qwen_finetuned_backend import load_finetuned_model

    config = json.loads(config_path.read_text())
    backend_config.set_data_dir(Path(config["data_dir"]))
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    corpus, candidate, base = (Path(config[key]) for key in ("corpus", "candidate", "base"))
    candidate_manifest = json.loads((candidate / "voicebox_finetune.json").read_text())
    cases = evaluation_cases(corpus, config["split"], config.get("novel"))
    reference = select_reference(read_jsonl(corpus / "train_verified.jsonl"))
    baseline_voice = local_baseline_voice(Path(config["data_dir"]), config["baseline_profile_id"])
    if baseline_voice is None:
        baseline_path = base
        old_audio, old_text = baseline_reference(Path(config["data_dir"]), config["baseline_profile_id"], output)
    else:
        baseline_path = Path(baseline_voice["model_path"])
        old_audio = old_text = None
    identity = {
        "config": config,
        "candidate_sha256": sha256(candidate / "model.safetensors"),
        "manifest_sha256": sha256(candidate / "voicebox_finetune.json"),
        "base_sha256": sha256(base / "model.safetensors"),
        "reference_sha256": sha256(Path(reference["audio"])),
        "baseline_reference_sha256": sha256(old_audio) if old_audio is not None else None,
        "baseline_voice": baseline_voice,
        "baseline_model_sha256": sha256(baseline_path / "model.safetensors"),
        "baseline_text": old_text,
        "cases": cases,
        "mlx_audio": importlib.metadata.version("mlx-audio"),
        "evaluation_code_sha256": sha256(Path(__file__)),
        "inference_sources": inference_source_identity(),
    }
    if identity["candidate_sha256"] != candidate_manifest["files"]["model.safetensors"]:
        raise ValueError("Candidate model identity changed")
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Evaluation inputs changed; use a new output directory")
    atomic_json(identity_path, identity)
    results = []
    for variant, path in (("candidate", candidate), ("baseline", baseline_path)):
        print(f"Loading {variant} for evaluation", flush=True)
        local = variant == "candidate" or baseline_voice is not None
        model = load_finetuned_model(str(path)) if local else load_model(str(path))
        if model.speech_tokenizer is None or model.tokenizer is None:
            raise RuntimeError("Model tokenizers failed to load")
        conditioning = None
        if not local:
            for patch in (
                _apply_mlx_audio_qwen_dtype_backport,
                apply_qwen_icl_cache_backport,
                apply_qwen_reference_decode_backport,
            ):
                if not patch(model, identity["mlx_audio"]):
                    raise RuntimeError("The existing voice's pinned inference optimization could not be applied")
            conditioning = prepare_reference_conditioning(model, audio_path=str(old_audio), reference_text=old_text)

        speaker = (
            candidate_manifest["speaker"]
            if variant == "candidate" or baseline_voice is None
            else baseline_voice["speaker"]
        )
        synthesis_variant = "candidate" if local else "baseline"
        synthesize_sample(
            model,
            synthesis_variant,
            "Esta es una prueba breve para preparar la voz.",
            7,
            speaker,
            conditioning,
            old_text,
        )
        for case in cases:
            metadata_path = output / f"{variant}-{case['id']}.json"
            wav_path = output / f"{variant}-{case['id']}.wav"
            if metadata_path.exists():
                row = json.loads(metadata_path.read_text())
                if row["sha256"] != sha256(wav_path):
                    raise ValueError("Saved evaluation audio changed")
            else:
                audio, seconds, tokens = synthesize_sample(
                    model, synthesis_variant, case["text"], case["seed"], speaker, conditioning, old_text
                )
                limit = min(4096, max(75, len(model.tokenizer.encode(case["text"])) * 6))
                sf.write(wav_path, audio, 24000, subtype="FLOAT")
                row = {
                    **case,
                    "variant": variant,
                    "audio": str(wav_path),
                    "sha256": sha256(wav_path),
                    "seconds": seconds,
                    "duration": len(audio) / 24000,
                    "tokens": tokens,
                    "hit_token_limit": tokens is not None and tokens >= limit,
                    "peak": float(np.abs(audio).max()),
                }
                atomic_json(metadata_path, row)
            results.append(row)
            print(
                json.dumps(
                    {"variant": variant, "id": case["id"], "seconds": row["seconds"], "duration": row["duration"]}
                ),
                flush=True,
            )

        if not local:
            target_vector = speaker_embedding(model, Path(reference["audio"]))
            for row in results:
                row["speaker_cosine"] = float(np.dot(target_vector, speaker_embedding(model, Path(row["audio"]))))
        del model, conditioning
        gc.collect()
        mx.clear_cache()
    if baseline_voice is not None:
        # CustomVoice exports omit this encoder. Score both with the same frozen
        # base only after releasing the two generation models.
        model = load_model(str(base))
        target_vector = speaker_embedding(model, Path(reference["audio"]))
        for row in results:
            row["speaker_cosine"] = float(np.dot(target_vector, speaker_embedding(model, Path(row["audio"]))))
        del model
        gc.collect()
        mx.clear_cache()
    atomic_json(output / "generation-summary.json", {"identity": identity, "results": results})


def reliable_transcription(result: dict) -> bool:
    """Reject empty or repetitive decoding using Whisper's compression cutoff."""
    return bool(result["text"].strip() and result["segments"]) and all(
        math.isfinite(segment["compression_ratio"]) and segment["compression_ratio"] <= 2.4
        for segment in result["segments"]
    )


def transcribe_audio(audio: str, whisper: str) -> dict:
    """Retry an ASR decoding loop once, without timestamps or reference text."""
    import mlx_whisper

    attempts = []
    for without_timestamps in (False, True):
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=whisper,
            language="es",
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=without_timestamps,
            verbose=None,
        )
        attempts.append({"without_timestamps": without_timestamps, "result": result})
        if reliable_transcription(result):
            break
    return {"text": result["text"].strip(), "reliable": reliable_transcription(result), "attempts": attempts}


def transcribe(output: Path, whisper: Path) -> None:
    """Check actual generated audio with the independent local ASR model."""
    generation = json.loads((output / "generation-summary.json").read_text())
    code_hash = sha256(Path(__file__))
    if generation["identity"]["evaluation_code_sha256"] != code_hash:
        raise ValueError("Evaluation code changed; generate a new identity-bound comparison")
    model_hash = sha256(whisper / "weights.npz")
    scored = []
    for row in generation["results"]:
        path = output / f"asr-{row['variant']}-{row['id']}.json"
        if sha256(Path(row["audio"])) != row["sha256"]:
            raise ValueError("Generated audio changed before transcription")
        if path.exists():
            result = json.loads(path.read_text())
            if (
                result["audio_sha256"] != row["sha256"]
                or result["model_sha256"] != model_hash
                or result.get("evaluation_code_sha256") != code_hash
            ):
                raise ValueError("ASR evaluation identity changed")
        else:
            result = {
                **transcribe_audio(row["audio"], str(whisper)),
                "audio_sha256": row["sha256"],
                "model_sha256": model_hash,
                "evaluation_code_sha256": code_hash,
            }
            atomic_json(path, result)
        if result["reliable"] is not True or not reliable_transcription(result["attempts"][-1]["result"]):
            raise ValueError(f"Unresolved ASR repetition or empty transcription: {row['variant']}/{row['id']}")
        edits, words = word_error(row["text"], result["text"])
        scored.append(
            {**row, "transcribed": result["text"], "word_errors": edits, "words": words, "wer": edits / words}
        )
    summary = summarize(scored)
    report = {**summary, "identity": generation["identity"], "results": scored}
    atomic_json(output / "evaluation.json", report)
    print(json.dumps(summary, indent=2), flush=True)


def summarize(scored: list[dict]) -> dict:
    """Apply the same predeclared quality gates during scoring and installation."""
    aggregates = {}
    for variant in ("candidate", "baseline"):
        rows = [row for row in scored if row["variant"] == variant]
        aggregates[variant] = {
            "wer": sum(row["word_errors"] for row in rows) / sum(row["words"] for row in rows),
            "worst_wer": max(row["wer"] for row in rows),
            "speaker_cosine": statistics.mean(row["speaker_cosine"] for row in rows),
            "median_seconds": statistics.median(row["seconds"] for row in rows),
            "median_rtf": statistics.median(row["seconds"] / row["duration"] for row in rows),
            "token_limit_hits": sum(row["hit_token_limit"] for row in rows),
            "incomplete_endings": sum(not has_complete_ending(row["text"], row["transcribed"]) for row in rows),
        }
    candidate, baseline = aggregates["candidate"], aggregates["baseline"]
    gates = {
        "intelligible": candidate["wer"] <= 0.05 and candidate["worst_wer"] <= 0.20,
        "transcript_not_regressed": candidate["wer"] <= baseline["wer"] + 0.02,
        "speaker_not_regressed": candidate["speaker_cosine"] >= baseline["speaker_cosine"] - 0.03,
        "warm_speed_not_regressed": candidate["median_rtf"] <= baseline["median_rtf"] * 1.10,
        "finished_audio": candidate["token_limit_hits"] == 0,
        "complete_endings": candidate["incomplete_endings"] == 0,
    }
    return {
        "accepted": all(gates.values()),
        "gates": gates,
        "aggregates": aggregates,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("config", type=Path)
    asr = sub.add_parser("transcribe")
    asr.add_argument("output", type=Path)
    asr.add_argument("whisper", type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.config)
    else:
        transcribe(args.output, args.whisper)

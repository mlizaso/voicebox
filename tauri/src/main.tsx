import ReactDOM from 'react-dom/client';
import { AppRoot } from '@/AppRoot';
// Import CSS from app directory using alias so Tailwind can scan the source files
import '@/index.css';
import { tauriPlatform } from './platform';

ReactDOM.createRoot(document.getElementById('root')!).render(<AppRoot platform={tauriPlatform} />);

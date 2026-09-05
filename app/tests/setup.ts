// Install storage before any persisted store is imported, regardless of test order.
export const storedValues = new Map<string, string>();
globalThis.localStorage = {
  getItem: (key) => storedValues.get(key) ?? null,
  setItem: (key, value) => {
    storedValues.set(key, value);
  },
  removeItem: (key) => {
    storedValues.delete(key);
  },
  clear: () => storedValues.clear(),
  key: (index) => [...storedValues.keys()][index] ?? null,
  get length() {
    return storedValues.size;
  },
};

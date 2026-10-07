import i18next from 'i18next';

export const testI18n = i18next.createInstance();
await testI18n.init({
  lng: 'en',
  initAsync: false,
  resources: { en: { translation: {} } },
});

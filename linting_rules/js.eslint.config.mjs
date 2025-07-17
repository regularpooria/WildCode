import { defineConfig } from 'eslint/config';

export default defineConfig({
  languageOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
  },
  rules: {
    // disable all common rules that might block syntax-only
    'no-unused-vars': 'off',
    'no-undef': 'off',
  },
});
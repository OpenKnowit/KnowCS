import { defineConfig } from 'vitest/config'
import { localeSplit } from './scripts/vite-plugins.mjs'

export default defineConfig({
  plugins: [localeSplit()],
  test: {
    include: ['src/**/*.test.ts'],
  },
})

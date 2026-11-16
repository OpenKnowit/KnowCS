import { defineConfig } from 'vitest/config'

// The admin API's own tests (server/admin.test.mjs): each run starts the real server on a spare port with an empty
// data folder. Not part of `npm test`; run them with `npm run test:server` after touching server/.
export default defineConfig({
  test: {
    include: ['server/**/*.test.mjs'],
    testTimeout: 15_000,
    hookTimeout: 20_000,
  },
})

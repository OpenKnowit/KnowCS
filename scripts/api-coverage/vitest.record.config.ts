// Runs the test suite and writes every program that ran without error and printed something: REC_OUT=file
export default { test: { setupFiles: ['scripts/api-coverage/record.setup.ts'], include: ['src/**/*.test.ts'], pool: 'forks' } }

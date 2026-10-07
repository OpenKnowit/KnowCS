/** An input error the UI can translate: code → lab.errors.<code>, params → interpolation. message stays English for tests and logs. */
export class LabError extends Error {
  code: string
  params: Record<string, string | number>
  constructor(code: string, params: Record<string, string | number>, message: string) {
    super(message)
    this.code = code
    this.params = params
  }
}

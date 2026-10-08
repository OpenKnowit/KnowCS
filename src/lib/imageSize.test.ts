import { describe, expect, it } from 'vitest'
// @ts-expect-error -- plain JS build helper
import { imageSize } from '../../scripts/vite-plugins.mjs'

describe('imageSize (note images reserve their space)', () => {
  it('reads WebP sizes (checked against PIL)', () => {
    expect(imageSize('src/content/notes/images/image_01.webp')).toEqual({ width: 1200, height: 654 })
    expect(imageSize('src/content/notes/images/image_04.webp')).toEqual({ width: 720, height: 502 })
  })
  it('returns null for a missing file', () => {
    expect(imageSize('src/content/notes/images/nope.webp')).toBeNull()
  })
})

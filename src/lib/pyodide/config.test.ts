import { describe, expect, it } from 'vitest'
import { importsOf, packageOf, pyodideBase } from './config'

describe('pyodide config', () => {
  it('defaults to the pinned jsDelivr build and always ends the base with a slash', () => {
    expect(pyodideBase('')).toBe('https://cdn.jsdelivr.net/pyodide/v314.0.7/full/')
    expect(pyodideBase(undefined)).toMatch(/^https:\/\/cdn\.jsdelivr\.net\/pyodide\/v[\d.]+\/full\/$/)
    expect(pyodideBase(' https://cdn.example.com/pyodide/full ')).toBe('https://cdn.example.com/pyodide/full/')
  })
  it('finds top-level imports in every import form', () => {
    const code = 'import numpy as np\nimport matplotlib.pyplot as plt, os\nfrom sklearn.cluster import KMeans\n  import cv2  # indented\nx = "import nope"\n# import commented'
    expect(importsOf(code)).toEqual(['numpy', 'matplotlib', 'os', 'sklearn', 'cv2'])
  })
  it('maps import names to package names', () => {
    expect(packageOf('sklearn')).toBe('scikit-learn')
    expect(packageOf('cv2')).toBe('opencv-python')
    expect(packageOf('numpy')).toBe('numpy')
  })
})

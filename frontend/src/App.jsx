import { useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [result, setResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  const handleUpload = async (event) => {
    const file = event.target.files?.[0]
    if (!file) return

    setSelectedFile(file)
    setResult(null)
    setError('')
    setPreviewUrl(URL.createObjectURL(file))

    const formData = new FormData()
    formData.append('file', file)

    setIsLoading(true)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed')
      }

      setResult(data)
    } catch (err) {
      setError(err.message || 'Something went wrong')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen px-6 py-10 text-slate-100">
      <div className="mx-auto max-w-5xl">
        <header className="mb-8 text-center">
          <p className="mb-2 text-sm uppercase tracking-[0.3em] text-cyan-400">AI Security</p>
          <h1 className="text-4xl font-bold md:text-5xl">Deepfake Image Detection</h1>
        </header>

        <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="rounded-2xl border border-slate-700 bg-slate-900/80 p-6 shadow-2xl shadow-cyan-950/20">
            <label className="flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-cyan-500/60 bg-slate-950/50 p-10 text-center transition hover:border-cyan-400 hover:bg-slate-950">
              <input type="file" accept="image/*" className="hidden" onChange={handleUpload} />
              <span className="mb-3 text-5xl">⬆️</span>
              <span className="text-lg font-medium">Upload image for analysis</span>
              <span className="mt-2 text-sm text-slate-400">PNG, JPG, JPEG supported</span>
            </label>

            {previewUrl && (
              <div className="mt-6 overflow-hidden rounded-xl border border-slate-700 bg-slate-950">
                <img src={previewUrl} alt="Uploaded preview" className="h-80 w-full object-cover" />
              </div>
            )}
          </div>

          <div className="rounded-2xl border border-slate-700 bg-slate-900/80 p-6 shadow-2xl shadow-cyan-950/20">
            <h2 className="mb-5 text-2xl font-semibold">Prediction result</h2>

            {isLoading && (
              <div className="flex items-center gap-3 text-cyan-300">
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-cyan-400 border-t-transparent" />
                <span>Analyzing image...</span>
              </div>
            )}

            {error && (
              <div className="rounded-xl border border-red-500/40 bg-red-500/10 p-4 text-red-300">{error}</div>
            )}

            {result && (
              <div className="space-y-4">
                <div className="rounded-xl border border-cyan-500/40 bg-cyan-500/10 p-4">
                  <p className="text-sm text-cyan-300">Verdict</p>
                  <p className="mt-2 text-3xl font-bold capitalize">{result.prediction}</p>
                </div>

                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="rounded-xl border border-slate-700 bg-slate-950/70 p-4">
                    <p className="text-sm text-slate-400">Score</p>
                    <p className="mt-2 text-xl font-semibold">{result.score}</p>
                  </div>
                  <div className="rounded-xl border border-slate-700 bg-slate-950/70 p-4">
                    <p className="text-sm text-slate-400">Confidence</p>
                    <p className="mt-2 text-xl font-semibold">{result.confidence}</p>
                  </div>
                </div>
              </div>
            )}

            {!result && !isLoading && !error && (
              <p className="text-slate-400">No analysis has been performed yet.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

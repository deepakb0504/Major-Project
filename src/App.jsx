import { useState, useEffect, useRef } from 'react'
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE_URL || ''

export default function App() {
  const [analytics, setAnalytics] = useState({ output_video: null, heatmap: null, tracks: null, summary: null })
  const [uploading, setUploading] = useState(false)
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)
  const liveVideoRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const liveChunksRef = useRef([])
  const [liveReady, setLiveReady] = useState(false)
  const [liveRecording, setLiveRecording] = useState(false)
  const [livePreviewUrl, setLivePreviewUrl] = useState(null)

  const fetchLatest = () => {
    axios.get(`${API_BASE}/api/analytics/latest`)
      .then(({ data }) => setAnalytics(data))
      .catch(() => setAnalytics({ output_video: null, heatmap: null, tracks: null, summary: null }))
  }

  useEffect(() => {
    fetchLatest()
    const t = setInterval(fetchLatest, 5000)
    return () => clearInterval(t)
  }, [])

  useEffect(() => {
    if (!jobId) return
    const poll = () => {
      axios.get(`${API_BASE}/api/jobs/${jobId}`)
        .then(({ data }) => {
          setJobStatus(data)
          if (data.status === 'completed' || data.status === 'failed') {
            setJobId(null)
            fetchLatest()
          }
        })
        .catch(() => setJobId(null))
    }
    poll()
    const id = setInterval(poll, 1500)
    return () => clearInterval(id)
  }, [jobId])

  const onUpload = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setError(null)
    setUploading(true)
    const form = new FormData()
    form.append('file', file)
    axios.post(`${API_BASE}/api/upload`, form, { headers: { 'Content-Type': 'multipart/form-data' } })
      .then(({ data }) => {
        setJobId(data.job_id)
        setJobStatus({ status: 'processing' })
      })
      .catch((err) => {
        setError(err.response?.data?.error || err.message)
        setUploading(false)
      })
      .finally(() => {
        setUploading(false)
        if (fileInputRef.current) fileInputRef.current.value = ''
      })
  }

  const setupLiveCamera = async () => {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      if (liveVideoRef.current) {
        liveVideoRef.current.srcObject = stream
        await liveVideoRef.current.play()
      }
      setLiveReady(true)
    } catch (e) {
      setError(e?.message || 'Camera permission denied / not available')
      setLiveReady(false)
    }
  }

  const stopLiveCamera = () => {
    const el = liveVideoRef.current
    const stream = el?.srcObject
    if (stream && stream.getTracks) {
      stream.getTracks().forEach((t) => t.stop())
    }
    if (el) el.srcObject = null
    setLiveReady(false)
  }

  const startRecording = () => {
    setError(null)
    const el = liveVideoRef.current
    const stream = el?.srcObject
    if (!stream) {
      setError('Live camera is not started yet.')
      return
    }
    liveChunksRef.current = []
    const mr = new MediaRecorder(stream, { mimeType: 'video/webm' })
    mediaRecorderRef.current = mr
    mr.ondataavailable = (evt) => {
      if (evt.data && evt.data.size > 0) liveChunksRef.current.push(evt.data)
    }
    mr.onstop = async () => {
      const blob = new Blob(liveChunksRef.current, { type: 'video/webm' })
      const url = URL.createObjectURL(blob)
      setLivePreviewUrl(url)
      // Upload to backend like a normal file
      try {
        const form = new FormData()
        form.append('file', new File([blob], 'live.webm', { type: 'video/webm' }))
        const { data } = await axios.post(`${API_BASE}/api/upload`, form, { headers: { 'Content-Type': 'multipart/form-data' } })
        setJobId(data.job_id)
        setJobStatus({ status: 'processing' })
      } catch (err) {
        setError(err.response?.data?.error || err.message)
      } finally {
        setLiveRecording(false)
      }
    }
    mr.start(250)
    setLiveRecording(true)
    // Auto-stop after 10 seconds to keep uploads manageable
    setTimeout(() => {
      try {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
          mediaRecorderRef.current.stop()
        }
      } catch {
        // ignore
      }
    }, 10_000)
  }

  const stopRecording = () => {
    try {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop()
      }
    } catch {
      // ignore
    }
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 p-6">
      <header className="mb-8">
        <h1 className="text-2xl font-bold text-slate-100">Smart Shop Analytics</h1>
        <p className="text-slate-400 text-sm mt-1">
          Upload a video, or record a short live clip in the browser, then view tracking + outputs below.
        </p>
      </header>

      <section className="mb-8">
        <label className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded-lg cursor-pointer transition">
          <input
            ref={fileInputRef}
            type="file"
            accept=".mp4,.avi,.mov,.mkv,.webm"
            onChange={onUpload}
            disabled={uploading}
            className="sr-only"
          />
          {uploading ? 'Uploading…' : 'Upload video'}
        </label>
        {jobStatus && (
          <span className="ml-3 text-slate-400">
            Job: {jobStatus.status}
            {jobStatus.status === 'failed' && jobStatus.error && ` — ${jobStatus.error}`}
          </span>
        )}
        {error && <p className="mt-2 text-red-400 text-sm">{error}</p>}
      </section>

      <section className="mb-8 rounded-xl bg-slate-800/80 overflow-hidden border border-slate-700">
        <div className="px-4 py-3 border-b border-slate-700 flex items-center justify-between gap-3 flex-wrap">
          <h2 className="text-slate-200 font-semibold">Live Camera (record 10s → upload)</h2>
          <div className="flex gap-2">
            {!liveReady ? (
              <button
                onClick={setupLiveCamera}
                className="px-3 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 transition text-sm"
              >
                Start camera
              </button>
            ) : (
              <button
                onClick={stopLiveCamera}
                className="px-3 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 transition text-sm"
              >
                Stop camera
              </button>
            )}
            <button
              onClick={liveRecording ? stopRecording : startRecording}
              disabled={!liveReady}
              className={`px-3 py-2 rounded-lg transition text-sm ${liveRecording ? 'bg-rose-600 hover:bg-rose-500' : 'bg-indigo-600 hover:bg-indigo-500'} disabled:opacity-50`}
            >
              {liveRecording ? 'Stop recording' : 'Record 10s'}
            </button>
          </div>
        </div>
        <div className="p-4 grid gap-4 md:grid-cols-2">
          <div className="aspect-video bg-black rounded-lg overflow-hidden border border-slate-700 flex items-center justify-center">
            <video ref={liveVideoRef} className="w-full h-full object-contain" playsInline muted />
          </div>
          <div className="aspect-video bg-black rounded-lg overflow-hidden border border-slate-700 flex items-center justify-center">
            {livePreviewUrl ? (
              <video src={livePreviewUrl} className="w-full h-full object-contain" controls playsInline />
            ) : (
              <p className="text-slate-500 text-sm">Preview appears after recording stops.</p>
            )}
          </div>
        </div>
      </section>

      <section className="grid gap-6 md:grid-cols-2">
        <div className="rounded-xl bg-slate-800/80 overflow-hidden border border-slate-700">
          <h2 className="px-4 py-3 text-slate-200 font-semibold border-b border-slate-700">Output video</h2>
          <div className="aspect-video bg-black flex items-center justify-center min-h-[240px]">
            {analytics.output_video ? (
              <video
                key={analytics.output_video}
                src={analytics.output_video}
                controls
                className="w-full h-full object-contain"
                playsInline
              >
                Your browser does not support the video tag.
              </video>
            ) : (
              <p className="text-slate-500 text-sm">No output yet. Upload a video or run live capture.</p>
            )}
          </div>
        </div>
        <div className="rounded-xl bg-slate-800/80 overflow-hidden border border-slate-700">
          <h2 className="px-4 py-3 text-slate-200 font-semibold border-b border-slate-700">Heatmap</h2>
          <div className="aspect-square bg-slate-900 flex items-center justify-center min-h-[240px]">
            {analytics.heatmap ? (
              <img
                src={analytics.heatmap}
                alt="Heatmap"
                className="max-w-full max-h-full object-contain"
              />
            ) : (
              <p className="text-slate-500 text-sm">No heatmap yet.</p>
            )}
          </div>
        </div>
      </section>

      <section className="mt-6 rounded-xl bg-slate-800/80 overflow-hidden border border-slate-700">
        <h2 className="px-4 py-3 text-slate-200 font-semibold border-b border-slate-700">Tracking (people IDs + dwell time)</h2>
        <div className="p-4">
          {analytics.summary ? (
            <div className="text-slate-300 text-sm mb-4 grid gap-2 md:grid-cols-4">
              <div><span className="text-slate-400">People:</span> {analytics.summary.unique_people_estimate}</div>
              <div><span className="text-slate-400">Avg dwell:</span> {analytics.summary.avg_dwell_time_s?.toFixed?.(1)}s</div>
              <div><span className="text-slate-400">Total dwell:</span> {analytics.summary.total_dwell_time_s?.toFixed?.(1)}s</div>
              <div><span className="text-slate-400">Frames:</span> {analytics.summary.frames} @ {analytics.summary.fps?.toFixed?.(1)}fps</div>
            </div>
          ) : (
            <p className="text-slate-500 text-sm mb-4">No tracking summary yet.</p>
          )}

          {Array.isArray(analytics.tracks) && analytics.tracks.length > 0 ? (
            <div className="overflow-auto">
              <table className="w-full text-sm">
                <thead className="text-slate-400">
                  <tr className="border-b border-slate-700">
                    <th className="text-left py-2 pr-3">ID</th>
                    <th className="text-left py-2 pr-3">Dwell (s)</th>
                    <th className="text-left py-2 pr-3">First seen (s)</th>
                    <th className="text-left py-2 pr-3">Last seen (s)</th>
                    <th className="text-left py-2 pr-3">Frames</th>
                  </tr>
                </thead>
                <tbody className="text-slate-200">
                  {analytics.tracks
                    .slice()
                    .sort((a, b) => (b.dwell_time_s || 0) - (a.dwell_time_s || 0))
                    .map((t) => (
                      <tr key={t.id} className="border-b border-slate-800">
                        <td className="py-2 pr-3">{t.id}</td>
                        <td className="py-2 pr-3">{Number(t.dwell_time_s || 0).toFixed(1)}</td>
                        <td className="py-2 pr-3">{Number(t.first_seen_s || 0).toFixed(1)}</td>
                        <td className="py-2 pr-3">{Number(t.last_seen_s || 0).toFixed(1)}</td>
                        <td className="py-2 pr-3">{t.seen_frames}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-slate-500 text-sm">Tracking table will appear after an analysis completes.</p>
          )}
        </div>
      </section>
    </div>
  )
}

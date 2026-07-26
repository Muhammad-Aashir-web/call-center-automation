import AudioCapture from './components/AudioCapture'

function App() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-slate-50 px-6 py-12">
      <div className="w-full max-w-xl space-y-8 rounded-2xl border border-slate-200 bg-white p-8 text-center shadow-sm">
        <h1 className="text-3xl font-semibold tracking-tight text-slate-900 sm:text-4xl">
          Call Center Automation — Live Audio
        </h1>
        <AudioCapture />
      </div>
    </main>
  )
}

export default App
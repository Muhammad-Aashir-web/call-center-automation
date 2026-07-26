import { useState } from 'react'

function AudioCapture() {
    const [isRecording, setIsRecording] = useState(false)
    const [stream, setStream] = useState<MediaStream | null>(null)
    const [error, setError] = useState<string | null>(null)

    const startRecording = async () => {
        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true })
            setStream(mediaStream)
            setError(null)
            setIsRecording(true)
        } catch (err) {
            console.error('Failed to start recording', err)
            setError('Microphone access was denied. Please allow access and try again.')
        }
    }

    const stopRecording = () => {
        stream?.getTracks().forEach((track) => track.stop())
        setStream(null)
        setIsRecording(false)
    }

    const handleToggleRecording = async () => {
        if (isRecording) {
            stopRecording()
            return
        }

        await startRecording()
    }

    return (
        <div className="flex flex-col items-start gap-3">
            <div className="flex items-center gap-3">
                <button
                    type="button"
                    onClick={handleToggleRecording}
                    className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-slate-700"
                >
                    {isRecording ? 'Stop Recording' : 'Start Recording'}
                </button>

                {isRecording && <span className="h-3 w-3 rounded-full bg-red-500 animate-pulse" aria-hidden="true" />}
            </div>

            {error && <p className="text-sm text-red-600">{error}</p>}
        </div>
    )
}

export default AudioCapture

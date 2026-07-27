import { useRef, useState } from 'react'

function AudioCapture() {
    const [isRecording, setIsRecording] = useState(false)
    const [stream, setStream] = useState<MediaStream | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [transcript, setTranscript] = useState<string[]>([])
    const isRecordingRef = useRef(false)
    const websocketRef = useRef<WebSocket | null>(null)
    const mediaRecorderRef = useRef<MediaRecorder | null>(null)
    const recorderTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

    const startRecording = async () => {
        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true })
            setStream(mediaStream)
            setError(null)
            setIsRecording(true)
            isRecordingRef.current = true

            const websocket = new WebSocket('ws://localhost:8000/ws/transcribe')
            websocketRef.current = websocket

            websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data as string) as { text?: string }

                    if (data.text) {
                        setTranscript((currentTranscript) => [...currentTranscript, data.text ?? ''])
                    }
                } catch (err) {
                    console.error('Failed to parse transcription response', err)
                }
            }

            const startNewRecorderCycle = () => {
                const mediaRecorder = new MediaRecorder(mediaStream, { mimeType: 'audio/webm' })
                mediaRecorderRef.current = mediaRecorder

                mediaRecorder.ondataavailable = (event) => {
                    if (websocket.readyState === WebSocket.OPEN && event.data.size > 0) {
                        websocket.send(event.data)
                    }
                }

                mediaRecorder.onstop = () => {
                    if (isRecordingRef.current) {
                        startNewRecorderCycle()
                    }
                }

                mediaRecorder.start()

                recorderTimeoutRef.current = setTimeout(() => {
                    mediaRecorder.stop()
                }, 3000)
            }

            websocket.onopen = () => {
                startNewRecorderCycle()
            }
        } catch (err) {
            console.error('Failed to start recording', err)
            setError('Microphone access was denied. Please allow access and try again.')
        }
    }

    const stopRecording = () => {
        isRecordingRef.current = false
        if (recorderTimeoutRef.current !== null) {
            clearTimeout(recorderTimeoutRef.current)
            recorderTimeoutRef.current = null
        }
        mediaRecorderRef.current?.stop()
        websocketRef.current?.close()
        stream?.getTracks().forEach((track) => track.stop())
        setStream(null)
        mediaRecorderRef.current = null
        websocketRef.current = null
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

            <div className="max-h-40 w-full overflow-y-auto rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-left text-sm text-slate-700">
                {transcript.join(' ')}
            </div>

            {error && <p className="text-sm text-red-600">{error}</p>}
        </div>
    )
}

export default AudioCapture

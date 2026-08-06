import { useRef, useState } from 'react'

interface KbSuggestion {
    id: string
    title: string
    category: string
    last_updated: string
    document: string
    rrf_score: number
}

interface SuggestedResponse {
    suggested_response: string
    rationale: string
}

interface CallUpdate {
    transcript: string | null
    intent: string | null
    intent_confidence: number | null
    kb_suggestions: KbSuggestion[]
    suggested_response: SuggestedResponse | null
    escalation_risk: number | null
    escalation_alert: boolean | null
}

interface QAScore {
    agent_id: string
    dimension_scores: Record<string, number>
    overall_score: number
    coaching_flags: string[]
    rationale: string
}

interface FollowUpTask {
    description: string
    priority: 'low' | 'medium' | 'high'
    due_by: string
}

interface AfterCallSummary {
    call_summary: string
    resolution_status: 'resolved' | 'unresolved' | 'escalated' | 'unknown'
    commitments: string[]
    compliance_notes: string[]
    follow_up_tasks: FollowUpTask[]
    requires_human_review: boolean
    review_reason: string | null
}

const RUBRIC_DIMENSIONS = ['greeting', 'active_listening', 'compliance', 'resolution', 'tone', 'closing']

function riskColor(risk: number | null): { bar: string; text: string; label: string } {
    if (risk === null) return { bar: 'bg-slate-300', text: 'text-slate-500', label: 'No data' }
    if (risk < 0.25) return { bar: 'bg-emerald-500', text: 'text-emerald-700', label: 'Calm' }
    if (risk < 0.4) return { bar: 'bg-amber-400', text: 'text-amber-700', label: 'Elevated' }
    return { bar: 'bg-red-500', text: 'text-red-700', label: 'High risk' }
}

function scoreColor(score: number): { bar: string; text: string } {
    if (score >= 80) return { bar: 'bg-emerald-500', text: 'text-emerald-700' }
    if (score >= 50) return { bar: 'bg-amber-400', text: 'text-amber-700' }
    return { bar: 'bg-red-500', text: 'text-red-700' }
}

function formatDimensionLabel(key: string): string {
    return key
        .replace(/_/g, ' ')
        .replace(/\b\w/g, (c) => c.toUpperCase())
}

function resolutionStatusStyle(status: AfterCallSummary['resolution_status']): string {
    switch (status) {
        case 'resolved':
            return 'bg-emerald-100 text-emerald-800'
        case 'unresolved':
            return 'bg-amber-100 text-amber-800'
        case 'escalated':
            return 'bg-red-100 text-red-800'
        default:
            return 'bg-slate-100 text-slate-600'
    }
}

function priorityStyle(priority: FollowUpTask['priority']): string {
    switch (priority) {
        case 'high':
            return 'bg-red-100 text-red-700'
        case 'medium':
            return 'bg-amber-100 text-amber-700'
        default:
            return 'bg-slate-100 text-slate-600'
    }
}

function CallDashboard() {
    const [callId, setCallId] = useState<string | null>(null)
    const [isRecording, setIsRecording] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [update, setUpdate] = useState<CallUpdate | null>(null)
    const [expandedKbId, setExpandedKbId] = useState<string | null>(null)

    const [qaScore, setQaScore] = useState<QAScore | null>(null)
    const [qaLoading, setQaLoading] = useState(false)
    const [qaError, setQaError] = useState<string | null>(null)

    const [summary, setSummary] = useState<AfterCallSummary | null>(null)
    const [summaryLoading, setSummaryLoading] = useState(false)
    const [summaryError, setSummaryError] = useState<string | null>(null)

    const isRecordingRef = useRef(false)
    const streamRef = useRef<MediaStream | null>(null)
    const websocketRef = useRef<WebSocket | null>(null)
    const mediaRecorderRef = useRef<MediaRecorder | null>(null)
    const recorderTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

    const startCall = async () => {
        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                },
            })
            streamRef.current = mediaStream
            setError(null)
            setUpdate(null)
            setQaScore(null)
            setQaError(null)
            setSummary(null)
            setSummaryError(null)

            const newCallId = crypto.randomUUID()
            setCallId(newCallId)
            setIsRecording(true)
            isRecordingRef.current = true

            const websocket = new WebSocket(`ws://localhost:8000/ws/calls/${newCallId}`)
            websocketRef.current = websocket

            websocket.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data as string) as {
                        type: 'update' | 'error'
                        data?: CallUpdate
                        message?: string
                    }

                    if (message.type === 'update' && message.data) {
                        setUpdate(message.data)
                    } else if (message.type === 'error') {
                        console.error('Call turn error:', message.message)
                    }
                } catch (err) {
                    console.error('Failed to parse call update', err)
                }
            }

            websocket.onerror = () => {
                setError('Connection to the call pipeline was lost.')
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
            console.error('Failed to start call', err)
            setError('Microphone access was denied. Please allow access and try again.')
        }
    }

    const endCall = () => {
        isRecordingRef.current = false
        if (recorderTimeoutRef.current !== null) {
            clearTimeout(recorderTimeoutRef.current)
            recorderTimeoutRef.current = null
        }
        mediaRecorderRef.current?.stop()
        websocketRef.current?.close()
        streamRef.current?.getTracks().forEach((track) => track.stop())
        streamRef.current = null
        mediaRecorderRef.current = null
        websocketRef.current = null
        setIsRecording(false)
    }

    const handleToggle = async () => {
        if (isRecording) {
            endCall()
            return
        }
        await startCall()
    }

    const fetchQaScore = async () => {
        if (!callId) return
        setQaLoading(true)
        setQaError(null)
        try {
            const response = await fetch(`http://localhost:8000/calls/${callId}/qa_score`)

            if (response.status === 404) {
                setQaError('No transcript found for this call yet.')
                return
            }
            if (!response.ok) {
                setQaError('Failed to fetch QA score. Please try again.')
                return
            }

            const data = (await response.json()) as QAScore
            setQaScore(data)
        } catch (err) {
            console.error('Failed to fetch QA score', err)
            setQaError('Could not reach the QA scoring service.')
        } finally {
            setQaLoading(false)
        }
    }

    const fetchSummary = async () => {
        if (!callId) return
        setSummaryLoading(true)
        setSummaryError(null)
        try {
            const response = await fetch(`http://localhost:8000/calls/${callId}/after_call_summary`)

            if (response.status === 404) {
                setSummaryError('No transcript found for this call yet.')
                return
            }
            if (!response.ok) {
                setSummaryError('Failed to generate the call summary. Please try again.')
                return
            }

            const data = (await response.json()) as AfterCallSummary
            setSummary(data)
        } catch (err) {
            console.error('Failed to fetch after-call summary', err)
            setSummaryError('Could not reach the after-call work service.')
        } finally {
            setSummaryLoading(false)
        }
    }

    const risk = riskColor(update?.escalation_risk ?? null)
    const riskPercent = Math.round((update?.escalation_risk ?? 0) * 100)

    const showPostCallSections = callId !== null && !isRecording

    return (
        <div className="mx-auto w-full max-w-6xl space-y-4 px-4 py-8">
            <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                    <h1 className="text-2xl font-semibold tracking-tight text-slate-900">
                        Live Call Console
                    </h1>
                    {callId && (
                        <p className="mt-1 font-mono text-xs text-slate-400">call_id: {callId}</p>
                    )}
                </div>
                <div className="flex items-center gap-3">
                    {isRecording && (
                        <span className="h-2.5 w-2.5 rounded-full bg-red-500 animate-pulse" aria-hidden="true" />
                    )}
                    <button
                        type="button"
                        onClick={handleToggle}
                        className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-slate-700"
                    >
                        {isRecording ? 'End Call' : 'Start Call'}
                    </button>
                </div>
            </div>

            {error && <p className="text-sm text-red-600">{error}</p>}

            {update?.escalation_alert && (
                <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-800">
                    Escalation risk alert — this call may need supervisor attention.
                </div>
            )}

            <div className="grid grid-cols-1 gap-4 lg:grid-cols-5">
                {/* Left column: transcript */}
                <div className="lg:col-span-3 space-y-4">
                    <div className="rounded-2xl border border-slate-200 bg-white p-5">
                        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">
                            Transcript
                        </h2>
                        <div className="max-h-72 overflow-y-auto rounded-lg bg-slate-50 p-4 font-mono text-sm leading-relaxed text-slate-700">
                            {update?.transcript || (
                                <span className="text-slate-400">Waiting for audio…</span>
                            )}
                        </div>
                    </div>

                    <div className="rounded-2xl border border-slate-200 bg-white p-5">
                        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">
                            Suggested Response
                        </h2>
                        {update?.suggested_response ? (
                            <div className="space-y-2">
                                <p className="rounded-lg bg-indigo-50 p-4 text-sm text-slate-800">
                                    {update.suggested_response.suggested_response}
                                </p>
                                <p className="text-xs text-slate-400">
                                    {update.suggested_response.rationale}
                                </p>
                            </div>
                        ) : (
                            <p className="text-sm text-slate-400">No suggestion yet.</p>
                        )}
                    </div>
                </div>

                {/* Right column: agent assist panel */}
                <div className="lg:col-span-2 space-y-4">
                    <div className="rounded-2xl border border-slate-200 bg-white p-5">
                        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">
                            Intent
                        </h2>
                        {update?.intent ? (
                            <div className="flex items-center justify-between">
                                <span className="rounded-full bg-slate-900 px-3 py-1 text-xs font-medium text-white">
                                    {update.intent}
                                </span>
                                <span className="font-mono text-xs text-slate-400">
                                    {update.intent_confidence !== null
                                        ? `${Math.round((update.intent_confidence ?? 0) * 100)}% confidence`
                                        : ''}
                                </span>
                            </div>
                        ) : (
                            <p className="text-sm text-slate-400">Not classified yet.</p>
                        )}
                    </div>

                    <div className="rounded-2xl border border-slate-200 bg-white p-5">
                        <div className="mb-2 flex items-center justify-between">
                            <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                                Escalation Risk
                            </h2>
                            <span className={`text-xs font-medium ${risk.text}`}>{risk.label}</span>
                        </div>
                        <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
                            <div
                                className={`h-full rounded-full transition-all duration-500 ${risk.bar}`}
                                style={{ width: `${riskPercent}%` }}
                            />
                        </div>
                        <p className="mt-1 font-mono text-xs text-slate-400">{riskPercent}%</p>
                    </div>

                    <div className="rounded-2xl border border-slate-200 bg-white p-5">
                        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">
                            Knowledge Base Matches
                        </h2>
                        {update?.kb_suggestions && update.kb_suggestions.length > 0 ? (
                            <ul className="space-y-2">
                                {update.kb_suggestions.map((kb) => {
                                    const isExpanded = expandedKbId === kb.id
                                    return (
                                        <li key={kb.id} className="rounded-lg border border-slate-100">
                                            <button
                                                type="button"
                                                onClick={() =>
                                                    setExpandedKbId(isExpanded ? null : kb.id)
                                                }
                                                className="flex w-full items-center justify-between px-3 py-2 text-left text-sm text-slate-700 hover:bg-slate-50"
                                            >
                                                <span>{kb.title}</span>
                                                <span className="ml-2 shrink-0 rounded-full bg-slate-100 px-2 py-0.5 text-[10px] uppercase tracking-wide text-slate-500">
                                                    {kb.category}
                                                </span>
                                            </button>
                                            {isExpanded && (
                                                <p className="whitespace-pre-line border-t border-slate-100 px-3 py-2 text-xs text-slate-500">
                                                    {kb.document}
                                                </p>
                                            )}
                                        </li>
                                    )
                                })}
                            </ul>
                        ) : (
                            <p className="text-sm text-slate-400">No matches yet.</p>
                        )}
                    </div>
                </div>
            </div>

            {/* QA score section — appears once a call has ended */}
            {showPostCallSections && (
                <div className="rounded-2xl border border-slate-200 bg-white p-5">
                    <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                            Quality Assurance Score
                        </h2>
                        <button
                            type="button"
                            onClick={fetchQaScore}
                            disabled={qaLoading}
                            className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-slate-300"
                        >
                            {qaLoading ? 'Scoring call…' : qaScore ? 'Re-score Call' : 'Get QA Score'}
                        </button>
                    </div>

                    {qaError && (
                        <p className="text-sm text-red-600">{qaError}</p>
                    )}

                    {!qaScore && !qaLoading && !qaError && (
                        <p className="text-sm text-slate-400">
                            Call ended. Click "Get QA Score" to score this call against the QA rubric.
                        </p>
                    )}

                    {qaScore && (
                        <div className="space-y-5">
                            <div className="flex items-center gap-3">
                                <span className={`text-3xl font-semibold ${scoreColor(qaScore.overall_score).text}`}>
                                    {qaScore.overall_score}
                                </span>
                                <span className="text-sm text-slate-400">/ 100 overall</span>
                            </div>

                            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                                {RUBRIC_DIMENSIONS.map((dimension) => {
                                    const score = qaScore.dimension_scores[dimension] ?? 0
                                    const colors = scoreColor(score)
                                    return (
                                        <div key={dimension}>
                                            <div className="mb-1 flex items-center justify-between">
                                                <span className="text-xs font-medium text-slate-600">
                                                    {formatDimensionLabel(dimension)}
                                                </span>
                                                <span className={`font-mono text-xs ${colors.text}`}>
                                                    {score}
                                                </span>
                                            </div>
                                            <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
                                                <div
                                                    className={`h-full rounded-full transition-all duration-500 ${colors.bar}`}
                                                    style={{ width: `${score}%` }}
                                                />
                                            </div>
                                        </div>
                                    )
                                })}
                            </div>

                            {qaScore.coaching_flags.length > 0 && (
                                <div>
                                    <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                                        Coaching Flags
                                    </h3>
                                    <ul className="space-y-1.5">
                                        {qaScore.coaching_flags.map((flag, idx) => (
                                            <li
                                                key={idx}
                                                className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800"
                                            >
                                                {flag}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            <div>
                                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                                    Rationale
                                </h3>
                                <p className="rounded-lg bg-slate-50 p-4 text-sm leading-relaxed text-slate-700">
                                    {qaScore.rationale}
                                </p>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* After-call summary section — appears once a call has ended */}
            {showPostCallSections && (
                <div className="rounded-2xl border border-slate-200 bg-white p-5">
                    <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                            After-Call Summary
                        </h2>
                        <button
                            type="button"
                            onClick={fetchSummary}
                            disabled={summaryLoading}
                            className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-slate-300"
                        >
                            {summaryLoading ? 'Summarizing…' : summary ? 'Regenerate Summary' : 'Get Call Summary'}
                        </button>
                    </div>

                    {summaryError && (
                        <p className="text-sm text-red-600">{summaryError}</p>
                    )}

                    {!summary && !summaryLoading && !summaryError && (
                        <p className="text-sm text-slate-400">
                            Call ended. Click "Get Call Summary" to generate a summary, CRM update, and follow-up tasks.
                        </p>
                    )}

                    {summary && (
                        <div className="space-y-5">
                            <div className="flex flex-wrap items-center gap-3">
                                <span
                                    className={`rounded-full px-3 py-1 text-xs font-medium capitalize ${resolutionStatusStyle(
                                        summary.resolution_status
                                    )}`}
                                >
                                    {summary.resolution_status}
                                </span>
                                {summary.requires_human_review && (
                                    <span className="rounded-full bg-red-100 px-3 py-1 text-xs font-medium text-red-800">
                                        Flagged for human review
                                    </span>
                                )}
                            </div>

                            <div>
                                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                                    Call Summary
                                </h3>
                                <p className="rounded-lg bg-slate-50 p-4 text-sm leading-relaxed text-slate-700">
                                    {summary.call_summary || 'No summary could be generated for this call.'}
                                </p>
                            </div>

                            {summary.requires_human_review && summary.review_reason && (
                                <p className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
                                    {summary.review_reason}
                                </p>
                            )}

                            {summary.commitments.length > 0 && (
                                <div>
                                    <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                                        Commitments Made
                                    </h3>
                                    <ul className="space-y-1.5">
                                        {summary.commitments.map((item, idx) => (
                                            <li
                                                key={idx}
                                                className="rounded-lg bg-indigo-50 px-3 py-2 text-xs text-indigo-800"
                                            >
                                                {item}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            {summary.compliance_notes.length > 0 && (
                                <div>
                                    <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                                        Compliance Notes
                                    </h3>
                                    <ul className="space-y-1.5">
                                        {summary.compliance_notes.map((item, idx) => (
                                            <li
                                                key={idx}
                                                className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800"
                                            >
                                                {item}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            {summary.follow_up_tasks.length > 0 && (
                                <div>
                                    <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                                        Follow-Up Tasks
                                    </h3>
                                    <ul className="space-y-2">
                                        {summary.follow_up_tasks.map((task, idx) => (
                                            <li
                                                key={idx}
                                                className="flex items-center justify-between rounded-lg border border-slate-100 px-3 py-2 text-sm"
                                            >
                                                <div>
                                                    <p className="text-slate-700">{task.description}</p>
                                                    <p className="mt-0.5 text-xs text-slate-400">Due {task.due_by}</p>
                                                </div>
                                                <span
                                                    className={`ml-3 shrink-0 rounded-full px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide ${priorityStyle(
                                                        task.priority
                                                    )}`}
                                                >
                                                    {task.priority}
                                                </span>
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}

export default CallDashboard
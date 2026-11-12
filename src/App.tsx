import { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  AlertCircle,
  ArrowDownToLine,
  Camera,
  Check,
  ChevronRight,
  CircleHelp,
  Cpu,
  Database,
  GraduationCap,
  Hand,
  LoaderCircle,
  Play,
  RotateCcw,
  Trash2,
  VideoOff,
  X,
} from "lucide-react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { composeHangul } from "./hangul";

type Point = { x: number; y: number; z: number };
type SampleSummary = { total: number; by_label: Record<string, number> };
type Sample = { id: string; label: string; created_at?: string };
type Prediction = { label: string; confidence: number };
type ClassMetric = { label: string; precision: number; recall: number; f1_score: number; support: number };
type TrainReport = {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  labels: string[];
  per_class: ClassMetric[];
  confusion_matrix: number[][];
  train_size: number;
  test_size: number;
  metric_caveat: string;
};

const consonants = ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"];
const vowels = ["ㅏ", "ㅑ", "ㅓ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ", "ㅐ", "ㅔ"];
const letters = [...consonants, ...vowels];
const connections: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const API = "/api";

class ApiError extends Error {
  constructor(message: string, readonly status: number, readonly code?: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => null) as {
      detail?: string | { message?: string; code?: string };
    } | null;
    const detail = body?.detail;
    const message = typeof detail === "string" ? detail : detail?.message;
    const code = typeof detail === "object" ? detail?.code : undefined;
    throw new ApiError(message ?? `Request failed (${response.status})`, response.status, code);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

function App() {
  const [page, setPage] = useState<"recognize" | "collect" | "train" | "learn">("recognize");
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [visionReady, setVisionReady] = useState(false);
  const [hasHand, setHasHand] = useState(false);
  const [selected, setSelected] = useState("ㄱ");
  const [stablePrediction, setStablePrediction] = useState<Prediction | null>(null);
  const [letterBuffer, setLetterBuffer] = useState<string[]>([]);
  const [autoCommit, setAutoCommit] = useState(true);
  const [threshold, setThreshold] = useState(0.65);
  const [summary, setSummary] = useState<SampleSummary>({ total: 0, by_label: {} });
  const [samples, setSamples] = useState<Sample[]>([]);
  const [collecting, setCollecting] = useState(false);
  const [message, setMessage] = useState("");
  const [working, setWorking] = useState(false);
  const [report, setReport] = useState<TrainReport | null>(null);
  const [modelReady, setModelReady] = useState(false);
  const [apiReady, setApiReady] = useState<boolean | null>(null);
  const [helpOpen, setHelpOpen] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const lastFrameRef = useRef(0);
  const lastPredictionAt = useRef(0);
  const lastCaptureAt = useRef(0);
  const captureSessionId = useRef<string | null>(null);
  const lastDuplicateNoticeAt = useRef(0);
  const predictionWindow = useRef<Prediction[]>([]);
  const candidateLabel = useRef<string | null>(null);
  const candidateSince = useRef(0);
  const committedLabel = useRef<string | null>(null);

  const refreshSamples = useCallback(async () => {
    try {
      const [data, list] = await Promise.all([
        request<SampleSummary>("/samples/summary"),
        request<Sample[]>("/samples"),
      ]);
      setSummary(data);
      setSamples(list);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not load dataset.");
    }
  }, []);

  useEffect(() => {
    void request<{ status: string; model_ready?: boolean }>("/health")
      .then((health) => {
        setApiReady(true);
        setModelReady(Boolean(health.model_ready));
      })
      .catch(() => setApiReady(false));
    void refreshSamples();
  }, [refreshSamples]);

  useEffect(() => {
    if (page === "collect" || page === "train") void refreshSamples();
  }, [page, refreshSamples]);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOn(false);
    setHasHand(false);
    setCollecting(false);
    predictionWindow.current = [];
    candidateLabel.current = null;
    candidateSince.current = 0;
    committedLabel.current = null;
  }, []);

  const startCamera = useCallback(async () => {
    setCameraError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 720 } },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      if (!landmarkerRef.current) {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm",
        );
        landmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 1,
        });
      }
      setVisionReady(true);
      setCameraOn(true);
    } catch (error) {
      stopCamera();
      setCameraError(error instanceof Error ? error.message : "Camera or hand tracking could not start.");
    }
  }, [stopCamera]);

  useEffect(() => {
    if (!cameraOn) return;
    let frameId = 0;
    const tick = async (now: number) => {
      const video = videoRef.current;
      const landmarker = landmarkerRef.current;
      const canvas = canvasRef.current;
      if (!video || !landmarker || !canvas) {
        frameId = requestAnimationFrame(tick);
        return;
      }
      if (now - lastFrameRef.current >= 75 && video.readyState >= 2) {
        lastFrameRef.current = now;
        const result = landmarker.detectForVideo(video, now);
        const ctx = canvas.getContext("2d");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
        const points = result.landmarks[0] as Point[] | undefined;
        setHasHand(Boolean(points));
        if (ctx && points) {
          ctx.lineWidth = Math.max(2, canvas.width / 280);
          ctx.strokeStyle = "#80f2c3";
          ctx.fillStyle = "#ffffff";
          for (const [a, b] of connections) {
            ctx.beginPath();
            ctx.moveTo(points[a].x * canvas.width, points[a].y * canvas.height);
            ctx.lineTo(points[b].x * canvas.width, points[b].y * canvas.height);
            ctx.stroke();
          }
          for (const point of points) {
            ctx.beginPath();
            ctx.arc(point.x * canvas.width, point.y * canvas.height, Math.max(2.5, canvas.width / 240), 0, Math.PI * 2);
            ctx.fill();
          }
          if (page === "collect" && collecting && now - lastCaptureAt.current >= 800) {
            lastCaptureAt.current = now;
            void request<Sample>("/samples", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ label: selected, landmarks: points, session_id: captureSessionId.current }),
            }).then((sample) => {
              setSamples((current) => [...current, sample]);
              setModelReady(false);
              setReport(null);
              setSummary((current) => ({
                total: current.total + 1,
                by_label: { ...current.by_label, [selected]: (current.by_label[selected] ?? 0) + 1 },
              }));
            }).catch((error: unknown) => {
              if (error instanceof ApiError && error.code === "duplicate_sample") {
                if (Date.now() - lastDuplicateNoticeAt.current > 3000) {
                  lastDuplicateNoticeAt.current = Date.now();
                  setMessage("Near-duplicate skipped. Move your hand to capture a distinct sample.");
                }
                return;
              }
              setCollecting(false);
              setMessage(error instanceof Error ? error.message : "Could not save sample.");
            });
          }
          if (page === "recognize" && modelReady && now - lastPredictionAt.current >= 120) {
            lastPredictionAt.current = now;
            void request<Prediction>("/predict", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ landmarks: points }),
            }).then((result) => {
              predictionWindow.current = [...predictionWindow.current.slice(-6), result];
              const counts = new Map<string, number>();
              for (const item of predictionWindow.current) counts.set(item.label, (counts.get(item.label) ?? 0) + item.confidence);
              const best = [...counts.entries()].sort((a, b) => b[1] - a[1])[0];
              if (best) {
                const smoothed = predictionWindow.current.filter((item) => item.label === best[0]);
                const confidence = smoothed.reduce((total, item) => total + item.confidence, 0) / smoothed.length;
                const next = { label: best[0], confidence };
                setStablePrediction(next);
                const isStable = predictionWindow.current.length >= 5
                  && smoothed.length >= 4
                  && confidence >= threshold;
                if (!isStable) {
                  candidateLabel.current = null;
                  candidateSince.current = 0;
                  return;
                }
                if (candidateLabel.current !== next.label) {
                  candidateLabel.current = next.label;
                  candidateSince.current = now;
                }
                if (committedLabel.current && committedLabel.current !== next.label) {
                  committedLabel.current = null;
                }
                if (
                  autoCommit
                  && committedLabel.current !== next.label
                  && now - candidateSince.current >= 700
                ) {
                  setLetterBuffer((current) => [...current, next.label]);
                  committedLabel.current = next.label;
                  setMessage(`${next.label} added automatically`);
                }
              }
            }).catch(() => {
              setModelReady(false);
              setMessage("Model unavailable. Train a model first.");
            });
          }
        } else {
          predictionWindow.current = [];
          setStablePrediction(null);
          candidateLabel.current = null;
          candidateSince.current = 0;
          committedLabel.current = null;
        }
      }
      frameId = requestAnimationFrame(tick);
    };
    frameId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameId);
  }, [autoCommit, cameraOn, collecting, modelReady, page, selected, threshold]);

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    landmarkerRef.current?.close();
  }, []);

  const confirmLetter = useCallback(() => {
    const guess = stablePrediction;
    if (!guess || guess.confidence < threshold) {
      setMessage("No confident stable prediction to add.");
      return;
    }
    setLetterBuffer((current) => [...current, guess.label]);
    committedLabel.current = guess.label;
    setMessage(`${guess.label} added`);
  }, [stablePrediction, threshold]);

  const deleteSample = async (id: string) => {
    try {
      await request(`/samples/${encodeURIComponent(id)}`, { method: "DELETE" });
      setModelReady(false);
      setReport(null);
      await refreshSamples();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not delete sample.");
    }
  };

  const exportCsv = async () => {
    try {
      const response = await fetch(`${API}/samples/export`);
      if (!response.ok) throw new Error(`CSV export failed (${response.status})`);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "korean_finger_alphabet.csv";
      link.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not export CSV.");
    }
  };

  const trainModel = async () => {
    setWorking(true);
    setMessage("");
    try {
      const result = await request<TrainReport>("/train", { method: "POST" });
      setReport(result);
      setModelReady(true);
      setMessage("Training complete. Model saved.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Training failed.");
    } finally {
      setWorking(false);
    }
  };

  const recognizedText = composeHangul(letterBuffer);
  const camera = (
    <div className="camera-shell">
      <video ref={videoRef} className={cameraOn ? "camera-video active" : "camera-video"} playsInline muted />
      <canvas ref={canvasRef} className="camera-overlay" />
      {!cameraOn && (
        <div className="camera-placeholder">
          <div className="camera-icon"><Camera size={28} /></div>
          <strong>카메라를 시작하세요 <span>Start your camera</span></strong>
          <p>Allow camera access to track your hand landmarks.</p>
        </div>
      )}
      {cameraOn && !hasHand && <div className="camera-hint"><Hand size={17} /> Show one hand in the frame</div>}
      {cameraOn && <div className="camera-live"><i /> LIVE · 13 FPS</div>}
      <div className="camera-controls">
        {!cameraOn
          ? <button className="button primary" onClick={() => void startCamera()}><Play size={16} /> Start camera</button>
          : <button className="button dark" onClick={stopCamera}><VideoOff size={16} /> Stop camera</button>}
      </div>
    </div>
  );

  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="#" onClick={(event) => event.preventDefault()}>
          <span className="brand-mark"><Hand size={19} /></span>
          <span>son<span className="brand-dot">.</span></span>
          <span className="brand-divider" />
          <span className="brand-sub">FINGER ALPHABET LAB</span>
        </a>
        <nav className="main-nav" aria-label="Main navigation">
          {([
            ["recognize", "Live Recognition", Activity],
            ["collect", "Data Collection", Database],
            ["train", "Model Training", Cpu],
            ["learn", "Learn Letters", GraduationCap],
          ] as const).map(([id, label, Icon]) => (
            <button key={id} className={page === id ? "nav-item selected" : "nav-item"} aria-label={label} title={label} onClick={() => setPage(id)}>
              <Icon size={16} /><span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="top-actions">
          <span className={apiReady ? "api-status online" : "api-status"}><i /> {apiReady ? "API connected" : apiReady === false ? "API offline" : "Connecting"}</span>
          <button className="icon-button" aria-label="About this project" onClick={() => setHelpOpen(true)}><CircleHelp size={19} /></button>
        </div>
      </header>

      <main className="main">
        <div className="page-heading">
          <div>
            <div className="eyebrow">KOREAN SIGN LANGUAGE · 지문자</div>
            <h1>{page === "recognize" ? "Finger Alphabet Recognition" : page === "collect" ? "Build your dataset" : page === "train" ? "Train your model" : "Learn the alphabet"}</h1>
            <p>{page === "recognize" ? "Practice Korean consonants and vowels with real-time hand tracking." : page === "collect" ? "Gather your own labeled landmarks. No sample data is preloaded." : page === "train" ? "Evaluate a Random Forest model from your collected samples." : "Explore the first-stage Korean fingerspelling letters."}</p>
          </div>
          <div className="heading-badge"><span className={modelReady ? "status-dot good" : "status-dot"} />{modelReady ? "Model ready" : "No trained model"}</div>
        </div>

        {cameraError && <div className="alert error"><AlertCircle size={18} />{cameraError}<button onClick={() => setCameraError("")}><X size={15} /></button></div>}
        {message && <div className="toast"><span>{message}</span><button onClick={() => setMessage("")} aria-label="Dismiss"><X size={15} /></button></div>}

        {page === "recognize" && (
          <div className="work-grid">
            <section className="panel camera-panel">
              <div className="panel-heading">
                <div><h2>Live camera</h2><p>MediaPipe hand landmark tracking</p></div>
                <span className={visionReady ? "soft-tag live-tag" : "soft-tag"}><span className={visionReady ? "status-dot good" : "status-dot"} />{visionReady ? "TRACKING READY" : "CAMERA OFF"}</span>
              </div>
              {camera}
              <div className="camera-footer">
                <span><span className={hasHand ? "tiny-check on" : "tiny-check"} />{hasHand ? "Hand detected · 21 landmarks" : "Waiting for a hand"}</span>
                <span>Threshold <b>{Math.round(threshold * 100)}%</b></span>
              </div>
            </section>
            <div className="right-column">
              <section className="panel prediction-panel">
                <div className="panel-heading compact"><div><h2>Current prediction</h2><p>Smoothed across recent frames</p></div><Activity size={18} className="muted-icon" /></div>
                <div className="prediction-display">
                  <div className="prediction-letter">{modelReady && stablePrediction && stablePrediction.confidence >= threshold ? stablePrediction.label : "—"}</div>
                  <div className="prediction-meta">
                    <span className="prediction-caption">PREDICTED LETTER</span>
                    <strong>{stablePrediction ? `${Math.round(stablePrediction.confidence * 100)}%` : "--"}</strong>
                    <span className="confidence-track"><i style={{ width: `${Math.min(100, (stablePrediction?.confidence ?? 0) * 100)}%` }} /></span>
                  </div>
                </div>
                <label className="threshold-control">Confidence threshold <span>{Math.round(threshold * 100)}%</span>
                  <input type="range" min="0.4" max="0.95" step="0.05" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} />
                </label>
                {!modelReady && <div className="inline-note"><AlertCircle size={15} /> Collect samples and train a model to enable recognition.</div>}
              </section>

              <section className="panel letters-panel">
                <div className="panel-heading compact"><div><h2>Alphabet</h2><p>Basic consonants · 자음 &nbsp; / &nbsp; vowels · 모음</p></div><button className="text-button" onClick={() => setPage("learn")}>Learn <ChevronRight size={14} /></button></div>
                <div className="letter-groups">
                  <div className="letter-row"><span className="group-label">자음</span><div className="letter-chips">{consonants.map((letter) => <button key={letter} onClick={() => setSelected(letter)} className={stablePrediction?.label === letter ? "letter-chip active" : "letter-chip"}>{letter}</button>)}</div></div>
                  <div className="letter-row"><span className="group-label">모음</span><div className="letter-chips">{vowels.map((letter) => <button key={letter} onClick={() => setSelected(letter)} className={stablePrediction?.label === letter ? "letter-chip active" : "letter-chip"}>{letter}</button>)}</div></div>
                </div>
              </section>

              <section className="panel text-panel">
                <div className="panel-heading compact"><div><h2>Your text</h2><p>{autoCommit ? "Auto-add after 0.7s stable · lift hand to repeat a letter" : "Confirm each letter to add it"}</p></div><span className="token-count">{letterBuffer.length} {letterBuffer.length === 1 ? "letter" : "letters"}</span></div>
                <div className={recognizedText ? "text-output has-text" : "text-output"}>
                  {recognizedText || <span className="text-placeholder">Your composed Korean text appears here…</span>}
                </div>
                <div className="jamo-strip">{letterBuffer.map((item, index) => <span key={`${index}-${item}`}>{item}</span>)}</div>
                <div className="text-actions">
                  <label className="auto-toggle">
                    <input type="checkbox" checked={autoCommit} onChange={(event) => {
                      setAutoCommit(event.target.checked);
                      candidateLabel.current = null;
                      candidateSince.current = 0;
                      committedLabel.current = null;
                    }} />
                    <span>Auto-add stable letters</span>
                  </label>
                  {!autoCommit && <button className="button primary" onClick={confirmLetter} disabled={!stablePrediction || stablePrediction.confidence < threshold}><Check size={15} /> Confirm letter</button>}
                  <button className="button secondary" onClick={() => setLetterBuffer((current) => current.slice(0, -1))} disabled={!letterBuffer.length}><RotateCcw size={15} /> Undo</button>
                  <button className="button ghost-danger" onClick={() => setLetterBuffer([])} disabled={!letterBuffer.length}><Trash2 size={15} /> Clear</button>
                </div>
              </section>
            </div>
          </div>
        )}

        {page === "collect" && (
          <div className="collect-grid">
            <section className="panel camera-panel">
              <div className="panel-heading"><div><h2>Capture samples</h2><p>Select a letter, show its hand shape, then collect</p></div><span className="soft-tag">21 LANDMARKS</span></div>
              {camera}
              <div className="capture-controls">
                <div><b>Capturing:</b> <span className="selected-label">{selected}</span> <span className="muted">· {summary.by_label[selected] ?? 0} samples</span></div>
                {!collecting
                  ? <button className="button primary" disabled={!cameraOn || !hasHand} onClick={() => { lastCaptureAt.current = 0; captureSessionId.current = crypto.randomUUID(); setCollecting(true); }}><Database size={16} /> Start capture</button>
                  : <button className="button danger" onClick={() => setCollecting(false)}><X size={16} /> Stop capture</button>}
              </div>
              {collecting && <div className="capture-notice"><i className="pulse-dot" /> Capturing every 800 ms · move your hand between samples</div>}
            </section>
            <section className="panel dataset-panel">
              <div className="panel-heading"><div><h2>Dataset</h2><p>{summary.total} labeled samples</p></div><button className="icon-button export-button" onClick={() => void exportCsv()} title="Export CSV"><ArrowDownToLine size={17} /></button></div>
              <div className="letter-groups collection-picker">
                <div className="letter-row"><span className="group-label">자음</span><div className="letter-chips">{consonants.map((letter) => <button key={letter} onClick={() => setSelected(letter)} className={selected === letter ? "letter-chip selected-letter" : "letter-chip"}>{letter}<small>{summary.by_label[letter] ?? 0}</small></button>)}</div></div>
                <div className="letter-row"><span className="group-label">모음</span><div className="letter-chips">{vowels.map((letter) => <button key={letter} onClick={() => setSelected(letter)} className={selected === letter ? "letter-chip selected-letter" : "letter-chip"}>{letter}<small>{summary.by_label[letter] ?? 0}</small></button>)}</div></div>
              </div>
              <div className="sample-list-heading"><span>RECENT SAMPLES</span><button onClick={() => void refreshSamples()}>Refresh</button></div>
              <div className="sample-list">
                {samples.slice().reverse().map((sample) => (
                  <div className="sample-item" key={sample.id}><span className="sample-letter">{sample.label}</span><code>{sample.id.slice(0, 12)}</code><button onClick={() => void deleteSample(sample.id)} aria-label={`Delete sample ${sample.id}`}><Trash2 size={14} /></button></div>
                ))}
                {!samples.length && <div className="empty-list">Samples you capture will appear here.</div>}
              </div>
              <div className="privacy-note"><AlertCircle size={15} /> Landmark coordinates only are stored; camera video is never uploaded.</div>
            </section>
          </div>
        )}

        {page === "train" && (
          <div className="train-layout">
            <section className="panel train-intro">
              <div className="train-icon"><Cpu size={24} /></div>
              <div><h2>Train a Random Forest classifier</h2><p>Training uses only the samples you collected. Samples are split into train and test sets; no synthetic duplication is performed.</p></div>
              <button className="button primary train-button" disabled={working || summary.total === 0} onClick={() => void trainModel()}>
                {working ? <><LoaderCircle size={16} className="spin" /> Training…</> : <><Play size={16} /> Train model</>}
              </button>
            </section>
            <div className="metrics-grid">
              <Metric title="Total samples" value={summary.total.toString()} note="Your labeled dataset" />
              <Metric title="Classes collected" value={`${Object.keys(summary.by_label).filter((key) => summary.by_label[key] > 0).length} / ${letters.length}`} note="Consonants + vowels" />
              <Metric title="Model status" value={modelReady ? "Ready" : "Not trained"} note={modelReady ? "Saved locally" : "Requires sufficient samples"} />
            </div>
            {report ? (
              <>
                <section className="panel report-panel">
                  <div className="panel-heading"><div><h2>Evaluation report</h2><p>Held-out test set · {report.train_size} train / {report.test_size} test samples</p></div><span className="soft-tag">TEST METRICS</span></div>
                  <div className="score-grid">
                    <Score label="Accuracy" value={report.accuracy} />
                    <Score label="Precision" value={report.precision} />
                    <Score label="Recall" value={report.recall} />
                    <Score label="F1 score" value={report.f1_score} />
                  </div>
                  <h3 className="table-title">Per-letter performance</h3>
                  <div className="table-scroll"><table><thead><tr><th>Letter</th><th>Precision</th><th>Recall</th><th>F1</th><th>Test samples</th></tr></thead><tbody>
                    {report.per_class.map((row) => {
                      const label = row.label;
                      return <tr key={label}><td className="table-letter">{label}</td><td>{(row.precision * 100).toFixed(1)}%</td><td>{(row.recall * 100).toFixed(1)}%</td><td>{(row.f1_score * 100).toFixed(1)}%</td><td>{row.support}</td></tr>;
                    })}
                  </tbody></table></div>
                  <div className="metric-caveat"><AlertCircle size={14} />{report.metric_caveat}</div>
                  <h3 className="table-title">Confusion matrix <span>rows = actual · columns = predicted</span></h3>
                  <div className="matrix-wrap"><table className="matrix"><thead><tr><th></th>{report.labels.map((label) => <th key={label}>{label}</th>)}</tr></thead><tbody>{report.confusion_matrix.map((row, i) => <tr key={report.labels[i]}><th>{report.labels[i]}</th>{row.map((value, j) => <td key={j} className={value ? (i === j ? "matrix-hit" : "matrix-miss") : ""}>{value || ""}</td>)}</tr>)}</tbody></table></div>
                </section>
              </>
            ) : (
              <section className="panel empty-report">
                <div className="empty-report-icon"><Activity size={24} /></div>
                <h2>No evaluation yet</h2>
                <p>Collect examples across the letters you want to recognize, then train to see honest held-out metrics here.</p>
              </section>
            )}
          </div>
        )}

        {page === "learn" && (
          <div className="learn-layout">
            <div className="learn-caveat">
              <AlertCircle size={18} />
              <div>
                <b>Use an authoritative Korean fingerspelling chart.</b>
                <p>The National Institute of Korean Language dictionary entries confirm the term and category, but do not provide a verified letter-by-letter pose chart here. We do not invent mappings or borrow ASL/JSL poses; verify a trusted Korean reference before labeling captures.</p>
                <div className="reference-links">
                  <a href="https://sldict.korean.go.kr/front/sign/signContentsView.do?origin_no=8087&top_category=CTE&category=&searchKeyword=%EC%A7%80%EB%AC%B8%EC%9E%90&searchCondition=&search_gubun=&museum_type=00&current_pos_index=0" target="_blank" rel="noreferrer">NIKL Korean Sign Language Dictionary · 지문자</a>
                  <a href="https://sldict.korean.go.kr/front/sign/signContentsView.do?origin_no=1383&top_category=CTE&category=&searchKeyword=%EC%A7%80%EB%AC%B8%EC%9E%90&searchCondition=&search_gubun=&museum_type=00&current_pos_index=1" target="_blank" rel="noreferrer">NIKL dictionary · 한글 지문자</a>
                </div>
              </div>
            </div>
            <section className="panel learn-panel">
              <div className="panel-heading"><div><h2>First-stage alphabet</h2><p>지문자 · Korean manual alphabet category set</p></div><span className="soft-tag">26 LETTERS</span></div>
              <h3 className="learn-group-title">자음 <span>Consonants</span></h3>
              <div className="learn-grid">{consonants.map((letter, i) => <button key={letter} className={selected === letter ? "learn-card chosen" : "learn-card"} onClick={() => setSelected(letter)}><span className="learn-letter">{letter}</span><span className="learn-index">CONSONANT {String(i + 1).padStart(2, "0")}</span><span className="learn-count">{summary.by_label[letter] ?? 0} samples</span></button>)}</div>
              <h3 className="learn-group-title">모음 <span>Vowels</span></h3>
              <div className="learn-grid">{vowels.map((letter, i) => <button key={letter} className={selected === letter ? "learn-card chosen" : "learn-card"} onClick={() => setSelected(letter)}><span className="learn-letter">{letter}</span><span className="learn-index">VOWEL {String(i + 1).padStart(2, "0")}</span><span className="learn-count">{summary.by_label[letter] ?? 0} samples</span></button>)}</div>
              <div className="learn-footer">Gesture poses are not illustrated here until verified against an authoritative Korean standard. Use the data collection page to label your captures only after checking a trusted reference.</div>
            </section>
          </div>
        )}

        <footer className="footer"><span>SON · KOREAN FINGER ALPHABET LAB</span><span>Local-first · Video stays in your browser</span></footer>
      </main>
      {helpOpen && <div className="modal-backdrop" onClick={() => setHelpOpen(false)}><section className="help-modal" onClick={(event) => event.stopPropagation()}><button className="modal-close" onClick={() => setHelpOpen(false)}><X size={18} /></button><div className="train-icon"><Hand size={22} /></div><h2>About Son</h2><p>A local-first Korean fingerspelling research and practice tool. It detects 21 hand landmarks in your browser and sends only coordinates to the local API.</p><div className="modal-note"><b>Important:</b> There is no bundled pretrained model or sample dataset. Recognition becomes available only after collecting labeled data and training a model.</div><button className="button primary" onClick={() => setHelpOpen(false)}>Got it</button></section></div>}
    </div>
  );
}

function Metric({ title, value, note }: { title: string; value: string; note: string }) {
  return <div className="panel metric-card"><span>{title}</span><strong>{value}</strong><small>{note}</small></div>;
}

function Score({ label, value }: { label: string; value: number }) {
  return <div className="score-card"><span>{label}</span><strong>{(value * 100).toFixed(1)}<small>%</small></strong><div className="score-bar"><i style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} /></div></div>;
}

export default App;

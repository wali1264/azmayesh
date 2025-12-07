import React, { useState, useEffect, useRef, createContext, useContext, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Modality } from "@google/genai";
import { 
  Beaker, 
  Microscope, 
  Activity, 
  Menu, 
  X, 
  MessageSquare, 
  Send, 
  Plus, 
  Info,
  AlertCircle,
  FlaskConical,
  Settings,
  ChevronLeft,
  Camera,
  Mic,
  MicOff,
  Video,
  VideoOff,
  Power,
  RefreshCw,
  Volume2,
  Upload,
  FileText,
  CheckCircle,
  AlertTriangle,
  Clock,
  Save,
  Trash2,
  User,
  History,
  Check,
  ChevronRight,
  Download,
  Thermometer,
  Heart,
  Wind,
  Stethoscope,
  Pill,
  Syringe,
  FilePlus,
  Languages,
  Brain,
  Wifi,
  WifiOff,
  Scale,
  Eye,
  EyeOff,
  Zap,
  Printer,
  Calculator,
  Grid,
  Scissors,
  Circle,
  Layers
} from 'lucide-react';

// --- TYPES & INTERFACES ---

type Tab = 'media-prep' | 'live-lab' | 'analysis' | 'settings' | 'resources';

interface Message {
  role: 'user' | 'model';
  text: string;
  timestamp: Date;
}

interface AnalysisResult {
  id: string;
  timestamp: number;
  organism_suspicion: string;
  confidence: string;
  growth_stage: string;
  colony_morphology: string;
  antibiotic_results: {
    name: string;
    zone_size_mm: number;
    interpretation: 'Sensitive' | 'Resistant' | 'Intermediate';
  }[];
  recommendation: string;
  imageUrl?: string;
}

interface UserProfile {
  name: string;
  labName: string;
  role: string;
}

interface VitalsData {
  temp: { value: number; enabled: boolean };
  hr: { value: number; enabled: boolean };
  bp: { sys: number; dia: number; enabled: boolean };
  spo2: { value: number; enabled: boolean };
  weight: { value: number; enabled: boolean };
}

interface ClinicalDiagnosisResult {
  diagnosis: string;
  reasoning: string;
  prescription_suggestion: string;
  media_recommendation: {
    standard: { name: string; description: string };
    emergency: { name: string; recipe: string };
  };
}

// --- AUDIO UTILS ---

const AUDIO_INPUT_SAMPLE_RATE = 16000;
const AUDIO_OUTPUT_SAMPLE_RATE = 24000;

function base64ToUint8Array(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function floatTo16BitPCM(input: Float32Array): Int16Array {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return output;
}

// --- KEY MANAGER & AI SERVICE ---

class KeyManager {
  private keys: string[] = [];
  private currentIndex = 0;
  private suspendedKeys = new Map<string, number>(); // Key -> Suspension End Timestamp
  private SUSPENSION_TIME = 60 * 1000; // 1 minute suspension on error

  constructor() {
    this.scanKeys();
  }

  private scanKeys() {
    const foundKeys: string[] = [];
    
    // 1. Add Primary Key (Safe check for process.env)
    // Note: In Vite, process.env might not be fully populated in the browser, 
    // but we check it just in case a define plugin is used.
    try {
      if (typeof process !== 'undefined' && process.env && process.env.API_KEY) {
        foundKeys.push(process.env.API_KEY);
      }
    } catch (e) {
      // Ignore process access errors
    }

    // 2. Scan VITE_GOOGLE_GENAI_TOKEN_1 to 20 using import.meta.env (Correct for Vite/Vercel)
    // We must list them explicitly for static analysis by the bundler.
    const env = import.meta.env;

    const potentialKeys = [
      env.VITE_GOOGLE_GENAI_TOKEN_1,
      env.VITE_GOOGLE_GENAI_TOKEN_2,
      env.VITE_GOOGLE_GENAI_TOKEN_3,
      env.VITE_GOOGLE_GENAI_TOKEN_4,
      env.VITE_GOOGLE_GENAI_TOKEN_5,
      env.VITE_GOOGLE_GENAI_TOKEN_6,
      env.VITE_GOOGLE_GENAI_TOKEN_7,
      env.VITE_GOOGLE_GENAI_TOKEN_8,
      env.VITE_GOOGLE_GENAI_TOKEN_9,
      env.VITE_GOOGLE_GENAI_TOKEN_10,
      env.VITE_GOOGLE_GENAI_TOKEN_11,
      env.VITE_GOOGLE_GENAI_TOKEN_12,
      env.VITE_GOOGLE_GENAI_TOKEN_13,
      env.VITE_GOOGLE_GENAI_TOKEN_14,
      env.VITE_GOOGLE_GENAI_TOKEN_15,
      env.VITE_GOOGLE_GENAI_TOKEN_16,
      env.VITE_GOOGLE_GENAI_TOKEN_17,
      env.VITE_GOOGLE_GENAI_TOKEN_18,
      env.VITE_GOOGLE_GENAI_TOKEN_19,
      env.VITE_GOOGLE_GENAI_TOKEN_20,
    ];

    potentialKeys.forEach(k => {
      if (k && k.trim().length > 0) {
        foundKeys.push(k);
      }
    });

    // Remove duplicates and empty strings
    this.keys = [...new Set(foundKeys)].filter(k => !!k && k.trim().length > 0);
    console.log(`[KeyManager] Initialized with ${this.keys.length} active keys.`);
  }

  public getAvailableKeyCount(): number {
    return this.keys.filter(k => !this.isSuspended(k)).length;
  }

  private isSuspended(key: string): boolean {
    if (!this.suspendedKeys.has(key)) return false;
    const endTime = this.suspendedKeys.get(key)!;
    if (Date.now() > endTime) {
      this.suspendedKeys.delete(key);
      return false;
    }
    return true;
  }

  public suspendKey(key: string) {
    console.warn(`[KeyManager] Suspending key ${key.substring(0, 8)}... for 1 minute.`);
    this.suspendedKeys.set(key, Date.now() + this.SUSPENSION_TIME);
  }

  public getNextKey(strategy: 'round-robin' | 'random' = 'round-robin'): string {
    if (this.keys.length === 0) {
      console.error("[KeyManager] No API keys available!");
      throw new Error("No API keys configuration found.");
    }

    const availableKeys = this.keys.filter(k => !this.isSuspended(k));
    
    // If all keys are suspended, use the one that expires soonest (force reuse)
    if (availableKeys.length === 0) {
      console.warn("[KeyManager] All keys suspended! Forcing reuse of primary key.");
      return this.keys[0]; 
    }

    if (strategy === 'random') {
      const randomIndex = Math.floor(Math.random() * availableKeys.length);
      return availableKeys[randomIndex];
    } else {
      // Round Robin
      const key = availableKeys[this.currentIndex % availableKeys.length];
      this.currentIndex++;
      return key;
    }
  }
}

class LabAIService {
  private keyManager: KeyManager;
  private modelName = 'gemini-2.5-flash';

  constructor() {
    this.keyManager = new KeyManager();
  }

  // Get a fresh client with a random key for Long-Lived Sessions (Live Lab)
  public getLiveClient(): GoogleGenAI {
    const key = this.keyManager.getNextKey('random');
    console.log(`[LabAIService] Starting Live Session with key: ${key.substring(0, 5)}...`);
    return new GoogleGenAI({ apiKey: key });
  }

  // Helper for One-Shot requests with Retry Logic (Round-Robin)
  private async executeWithRetry<T>(operation: (ai: GoogleGenAI) => Promise<T>): Promise<T> {
    const maxRetries = Math.max(3, this.keyManager.getAvailableKeyCount());
    let lastError: any;

    for (let i = 0; i < maxRetries; i++) {
      const key = this.keyManager.getNextKey('round-robin');
      try {
        const ai = new GoogleGenAI({ apiKey: key });
        return await operation(ai);
      } catch (error: any) {
        console.warn(`[LabAIService] Request failed with key ${key.substring(0, 5)}...`, error);
        lastError = error;
        
        // If it's a quota or permission error, suspend the key
        if (error.status === 429 || error.status === 403 || error.message?.includes('429')) {
          this.keyManager.suspendKey(key);
        } else {
          // If it's a network error unrelated to key, maybe break or retry? 
          // We continue to retry with another key just in case.
        }
      }
    }
    throw lastError || new Error("All API keys failed.");
  }

  async clinicalDiagnosis(
    vitals: VitalsData, 
    history: string, 
    images: string[]
  ): Promise<ClinicalDiagnosisResult | null> {
    return this.executeWithRetry(async (ai) => {
      // Build Vitals String
      let vitalsStr = "Vital Signs: ";
      if (vitals.temp.enabled) vitalsStr += `Temp: ${vitals.temp.value}°C, `;
      if (vitals.hr.enabled) vitalsStr += `Heart Rate: ${vitals.hr.value} bpm, `;
      if (vitals.bp.enabled) vitalsStr += `BP: ${vitals.bp.sys}/${vitals.bp.dia} mmHg, `;
      if (vitals.spo2.enabled) vitalsStr += `SpO2: ${vitals.spo2.value}%, `;
      if (vitals.weight.enabled) vitalsStr += `Weight: ${vitals.weight.value} kg, `;
      
      const promptParts: any[] = [
        { text: `Act as "Dr. Azam", a Chief Medical Officer and Microbiologist. 
        Analyze this patient data to provide a preliminary diagnosis and recommend the BEST culture media.
        
        ${vitalsStr}
        Clinical History: ${history}
        
        Task:
        1. Diagnose the potential infection based on symptoms and vitals.
        2. Suggest empirical treatment (Antibiotics).
        3. Recommend the BEST standard culture media for this pathogen.
        4. Provide an 'Emergency Homemade Media' recipe if standard media is unavailable (using common ingredients like eggs, starch, etc).
        
        Return ONLY valid JSON in this format:
        {
          "diagnosis": "Diagnosis in Persian",
          "reasoning": "Brief explanation in Persian",
          "prescription_suggestion": "Medicine names in Persian/English",
          "media_recommendation": {
            "standard": { "name": "Standard Media Name", "description": "Why this media?" },
            "emergency": { "name": "Homemade Option Name", "recipe": "Brief recipe instructions" }
          }
        }` }
      ];

      // Add images if any
      for (const img of images) {
        promptParts.push({ inlineData: { mimeType: "image/jpeg", data: img } });
      }

      const response = await ai.models.generateContent({
        model: this.modelName,
        contents: promptParts,
        config: { responseMimeType: 'application/json' }
      });

      return JSON.parse(response.text || '{}') as ClinicalDiagnosisResult;
    }).catch(err => {
      console.error("Diagnosis Fatal Error:", err);
      return null;
    });
  }

  async analyzePlateImage(base64Image: string): Promise<AnalysisResult | null> {
    return this.executeWithRetry(async (ai) => {
      const response = await ai.models.generateContent({
        model: this.modelName,
        contents: [
          {
            inlineData: {
              mimeType: "image/jpeg",
              data: base64Image
            }
          },
          {
            text: `Analyze this microbiology petri dish image carefully. You are an expert microbiologist.
            Identify colonies, morphology, and if antibiotic discs are present, estimate the Zone of Inhibition.
            
            Return the result ONLY as a JSON object with this exact structure:
            {
              "organism_suspicion": "Scientific name (e.g. Staphylococcus aureus)",
              "confidence": "High/Medium/Low",
              "growth_stage": "e.g. Early growth (4-6h) / Mature (24h)",
              "colony_morphology": "Brief description (color, shape, hemolysis)",
              "antibiotic_results": [
                { "name": "Antibiotic Code/Name", "zone_size_mm": number, "interpretation": "Sensitive/Resistant/Intermediate" }
              ],
              "recommendation": "Clinical suggestion in Persian (Farsi)"
            }
            Do not wrap in markdown code blocks. Just return the JSON string.`
          }
        ]
      });

      const text = response.text || '';
      const cleanJson = text.replace(/```json/g, '').replace(/```/g, '').trim();
      const parsed = JSON.parse(cleanJson);
      
      return {
        ...parsed,
        id: crypto.randomUUID(),
        timestamp: Date.now()
      } as AnalysisResult;
    }).catch(err => {
      console.error("Analysis Fatal Error:", err);
      return null;
    });
  }
}

const aiService = new LabAIService();

// --- COMPONENTS ---

// 1. Layout & Navigation
const SidebarItem = ({ icon: Icon, label, active, onClick }: any) => (
  <button
    onClick={onClick}
    className={`w-full flex items-center space-x-3 space-x-reverse px-4 py-3 rounded-xl transition-all duration-200 ${
      active 
        ? 'bg-blue-600 text-white shadow-lg shadow-blue-200' 
        : 'text-slate-600 hover:bg-slate-100'
    }`}
  >
    <Icon size={20} />
    <span className="font-medium">{label}</span>
  </button>
);

const MobileNavItem = ({ icon: Icon, label, active, onClick }: any) => (
  <button
    onClick={onClick}
    className={`flex flex-col items-center justify-center w-full py-2 space-y-1 ${
      active ? 'text-blue-600' : 'text-slate-500'
    }`}
  >
    <Icon size={24} strokeWidth={active ? 2.5 : 2} />
    <span className="text-[10px] font-medium">{label}</span>
  </button>
);

// 2. New Component: Live Voice Assistant (Updated for Key Rotation & single Context)
const LiveVoiceAssistant = ({ initialContext, persona = 'clinical' }: { initialContext: string, persona?: 'clinical' | 'qc' }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [volume, setVolume] = useState(0);
  const [messages, setMessages] = useState<Message[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Audio Refs - Use ONE shared AudioContext for Input and Output to prevent leaks
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const audioQueueRef = useRef<AudioBufferSourceNode[]>([]);
  const nextStartTimeRef = useRef<number>(0);
  const sessionRef = useRef<any>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  useEffect(scrollToBottom, [messages]);

  const initAudioContext = () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: AUDIO_OUTPUT_SAMPLE_RATE,
        latencyHint: 'interactive',
      });
    }
    if (audioContextRef.current.state === 'suspended') {
      audioContextRef.current.resume();
    }
  };

  const stopAllAudio = () => {
    audioQueueRef.current.forEach(source => {
      try { source.stop(); } catch(e) {}
    });
    audioQueueRef.current = [];
    if (audioContextRef.current) {
      nextStartTimeRef.current = audioContextRef.current.currentTime;
    }
  };

  const connect = async () => {
    try {
      setIsConnecting(true);
      initAudioContext();
      
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: AUDIO_INPUT_SAMPLE_RATE,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      streamRef.current = stream;

      const model = 'gemini-2.5-flash-native-audio-preview-09-2025';
      
      const historyText = messages.map(m => `${m.role === 'user' ? 'کاربر' : 'مدل'}: ${m.text}`).join('\n');
      const contextPrompt = `
      [CURRENT CONTEXT]
      ${initialContext}

      [CONVERSATION HISTORY]
      ${historyText}
      `;

      let systemInstruction = '';
      if (persona === 'clinical') {
        systemInstruction = `شما "دکتر اعظم" هستید، همکار متخصص و شوخ‌طبع. 
          1. **همکار:** مثل یک همکار صمیمی رفتار کنید.
          2. **شوخ:** خشک نباشید.
          3. **سریع:** اگر کاربر عجله دارد، کوتاه جواب دهید.
          4. **هوشمند:** اگر کاربر پرید وسط حرف، ساکت شوید.`;
      } else {
        systemInstruction = `شما "متخصص QC" هستید. تمرکز روی استانداردهای ساخت دیسک و استریلیزاسیون. دقیق و فنی باشید.`;
      }

      // Use a fresh client from the key pool
      const client = aiService.getLiveClient();

      const sessionPromise = client.live.connect({
        model: model,
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: `${systemInstruction}\n${contextPrompt}`,
          inputAudioTranscription: {}, 
          outputAudioTranscription: {}, 
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: persona === 'clinical' ? 'Kore' : 'Fenrir' } }
          }
        },
        callbacks: {
          onopen: () => {
            setIsConnecting(false);
            setIsConnected(true);
            sessionPromise.then(s => { sessionRef.current = s; });

            // Use the SHARED audio context for input to prevent "not enough resources" error
            const ctx = audioContextRef.current!;
            const source = ctx.createMediaStreamSource(stream);
            const processor = ctx.createScriptProcessor(4096, 1, 1);

            processor.onaudioprocess = (e) => {
              const inputData = e.inputBuffer.getChannelData(0);
              
              let sum = 0;
              for(let i=0; i<inputData.length; i+=50) sum += Math.abs(inputData[i]);
              setVolume(Math.min(100, (sum / (inputData.length/50)) * 500));

              const pcmData = floatTo16BitPCM(inputData);
              const uint8 = new Uint8Array(pcmData.buffer);
              const base64 = arrayBufferToBase64(uint8.buffer);

              sessionPromise.then(session => {
                session.sendRealtimeInput({
                  media: { mimeType: "audio/pcm;rate=16000", data: base64 }
                });
              });
            };

            source.connect(processor);
            processor.connect(ctx.destination);
            processorRef.current = processor;
            sourceRef.current = source;
          },
          onmessage: (msg) => {
            if (msg.serverContent?.interrupted) {
              stopAllAudio();
              return;
            }

            if (msg.serverContent?.outputTranscription?.text) {
               const text = msg.serverContent.outputTranscription.text;
               setMessages(prev => {
                  const last = prev[prev.length-1];
                  if (last && last.role === 'model' && (Date.now() - last.timestamp.getTime() < 5000)) {
                     return [...prev.slice(0, -1), { ...last, text: last.text + text }];
                  }
                  return [...prev, { role: 'model', text, timestamp: new Date() }];
               });
            }

            if (msg.serverContent?.inputTranscription?.text) {
               const text = msg.serverContent.inputTranscription.text;
               setMessages(prev => {
                  const last = prev[prev.length-1];
                  if (last && last.role === 'user' && (Date.now() - last.timestamp.getTime() < 5000)) {
                     return [...prev.slice(0, -1), { ...last, text: last.text + text }];
                  }
                  return [...prev, { role: 'user', text, timestamp: new Date() }];
               });
            }

            const audioData = msg.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (audioData && audioContextRef.current) {
              const ctx = audioContextRef.current;
              const audioBytes = base64ToUint8Array(audioData);
              const dataInt16 = new Int16Array(audioBytes.buffer);
              const float32 = new Float32Array(dataInt16.length);
              for (let i = 0; i < dataInt16.length; i++) {
                float32[i] = dataInt16[i] / 32768.0;
              }
              
              const buffer = ctx.createBuffer(1, float32.length, AUDIO_OUTPUT_SAMPLE_RATE);
              buffer.getChannelData(0).set(float32);
              
              const source = ctx.createBufferSource();
              source.buffer = buffer;
              source.connect(ctx.destination);
              
              const currentTime = ctx.currentTime;
              const startTime = Math.max(currentTime, nextStartTimeRef.current);
              source.start(startTime);
              nextStartTimeRef.current = startTime + buffer.duration;
              
              audioQueueRef.current.push(source);
              source.onended = () => {
                const index = audioQueueRef.current.indexOf(source);
                if (index > -1) audioQueueRef.current.splice(index, 1);
              };
            }
          },
          onclose: () => {
            setIsConnected(false);
          },
          onerror: (err) => {
            console.error(err);
            setIsConnected(false);
          }
        }
      });

    } catch (err) {
      console.error(err);
      setIsConnecting(false);
    }
  };

  const disconnect = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (processorRef.current && sourceRef.current) {
      processorRef.current.disconnect();
      sourceRef.current.disconnect();
    }
    stopAllAudio();
    setIsConnected(false);
    setIsConnecting(false);
    sessionRef.current = null;
    
    // Cleanup AudioContext if needed or leave for reuse
    // audioContextRef.current?.close(); 
    // audioContextRef.current = null;
  };

  return (
    <div className="flex flex-col h-[500px] bg-slate-900 text-white rounded-2xl shadow-xl overflow-hidden relative">
      {/* Header */}
      <div className="bg-slate-800 p-4 flex items-center justify-between border-b border-slate-700">
        <div className="flex items-center gap-3">
          <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500 shadow-[0_0_10px_#22c55e]' : 'bg-red-500'}`}></div>
          <div>
            <h3 className="font-bold text-sm">{persona === 'clinical' ? 'دکتر اعظم (بالینی)' : 'کارشناس کنترل کیفی (QC)'}</h3>
            <p className="text-[10px] text-slate-400">
              {persona === 'clinical' ? 'حالت: همکار هوشمند و سریع' : 'حالت: مشاور فنی ساخت محلول'}
            </p>
          </div>
        </div>
        {isConnected && (
           <div className="flex items-center gap-1">
             <div className="w-1 bg-green-500 animate-[bounce_1s_infinite]" style={{height: `${Math.max(4, volume/5)}px`}}></div>
             <div className="w-1 bg-green-500 animate-[bounce_1.2s_infinite]" style={{height: `${Math.max(6, volume/3)}px`}}></div>
             <div className="w-1 bg-green-500 animate-[bounce_0.8s_infinite]" style={{height: `${Math.max(4, volume/5)}px`}}></div>
           </div>
        )}
      </div>

      {/* Chat History / Visualizer Area */}
      <div className="flex-1 p-4 overflow-y-auto space-y-4 bg-gradient-to-b from-slate-900 to-slate-800">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-slate-500 opacity-60">
            <Wifi size={48} className="mb-4" />
            <p className="text-center text-sm px-6">
              برای شروع مکالمه {persona === 'clinical' ? 'بالینی' : 'فنی'}، دکمه میکروفون را فشار دهید.
              <br/>
              <span className="text-[10px] mt-2 block">
                 سیستم به صورت خودکار به بهترین سرور (کلید فعال) متصل می‌شود.
              </span>
            </p>
          </div>
        )}
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] rounded-2xl p-3 text-sm ${
              msg.role === 'user' 
              ? 'bg-blue-600 text-white rounded-tr-none' 
              : 'bg-slate-700 text-slate-200 rounded-tl-none'
            }`}>
              {msg.text}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Control Bar */}
      <div className="p-6 bg-slate-800 border-t border-slate-700 flex flex-col items-center justify-center relative">
        {isConnecting ? (
          <div className="flex flex-col items-center gap-2">
            <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
            <span className="text-xs text-blue-400 font-bold">در حال برقراری تماس ایمن...</span>
          </div>
        ) : (
          <button
            onClick={isConnected ? disconnect : connect}
            className={`w-16 h-16 rounded-full flex items-center justify-center transition-all transform hover:scale-105 shadow-2xl ${
              isConnected 
              ? 'bg-red-500 hover:bg-red-600 shadow-red-900/50 animate-pulse' 
              : 'bg-green-500 hover:bg-green-600 shadow-green-900/50'
            }`}
          >
            {isConnected ? <MicOff size={32} /> : <Mic size={32} />}
          </button>
        )}
        
        <p className="mt-4 text-xs font-medium text-slate-400">
          {isConnected ? 'مکالمه فعال است' : 'برای شروع مکالمه کلیک کنید'}
        </p>
      </div>
    </div>
  );
};

// ... [ClinicalConsoleModule omitted - stays same] ...
// ... [AnalysisModule omitted - stays same] ...
// ... [SettingsModule omitted - stays same] ...

// 7. Module: Lab Resources (Updated with Thermal Print Optimization)
const LabResourcesModule = () => {
  const [mode, setMode] = useState<'calculator' | 'designer'>('designer');
  const [showPreviewModal, setShowPreviewModal] = useState(false);
  
  // Calculator State
  const [calcSourceMg, setCalcSourceMg] = useState(500); 
  const [calcTargetMcg, setCalcTargetMcg] = useState(10); 
  const [calcDropUl, setCalcDropUl] = useState(20); 
  
  // Designer State
  const [printQueue, setPrintQueue] = useState<{code: string, name: string}[]>([]);
  const [discShape, setDiscShape] = useState<'circle' | 'square'>('circle');
  const [newDiscName, setNewDiscName] = useState('');
  const [newDiscCode, setNewDiscCode] = useState('');
  const [batchSize, setBatchSize] = useState(50); // Default batch size

  // Pre-defined antibiotics
  const commonAntibiotics = [
    { name: 'Amoxicillin', code: 'AMX' },
    { name: 'Ciprofloxacin', code: 'CIP' },
    { name: 'Gentamicin', code: 'CN' },
    { name: 'Erythromycin', code: 'E' },
    { name: 'Tetracycline', code: 'TE' },
    { name: 'Vancomycin', code: 'VA' },
  ];

  const requiredConc = calcTargetMcg / calcDropUl; 
  const requiredSolvent = calcSourceMg / requiredConc; 

  const addToQueue = (ab: {name: string, code: string}) => {
    // Add batch quantity
    const newItems = Array(batchSize).fill(ab);
    setPrintQueue(prev => [...prev, ...newItems]);
  };

  const handlePrint = () => {
    window.print();
  };

  const getCalculatorContext = () => {
    return `Current Calculator State:
    Source Antibiotic: ${calcSourceMg} mg
    Target Disc Potency: ${calcTargetMcg} mcg
    Pipette Drop Size: ${calcDropUl} ul
    Calculated Required Solvent: ${requiredSolvent.toFixed(2)} ml`;
  };

  return (
    <div className="space-y-6">
      <style>{`
        @media print {
          @page {
            size: 80mm auto; /* Force thermal roll width */
            margin: 0;
          }
          html, body {
            width: 80mm;
            margin: 0;
            padding: 0;
            background: white;
          }
          body * {
            visibility: hidden;
            display: none;
          }
          /* Show only modal and its children */
          #preview-modal {
            visibility: visible !important;
            display: block !important;
            position: absolute;
            top: 0;
            left: 0;
            width: 80mm !important;
            height: auto !important;
            background: white;
            z-index: 9999;
            overflow: visible;
          }
          #preview-modal * {
            visibility: visible !important;
            display: block; /* Default display, flex handled below if needed */
          }
          #printable-area-wrapper {
             display: flex !important;
             justify-content: center !important;
          }
          #printable-area {
            display: flex !important;
            flex-wrap: wrap !important;
            justify-content: flex-start !important;
            gap: 0 !important;
            width: 72mm !important; /* 80mm - margins */
            padding: 0 !important;
            margin: 0 auto !important;
            box-shadow: none !important;
            background: white !important;
          }
          .disc-preview {
            visibility: visible !important;
            display: flex !important;
            border: 1px solid black !important;
            color: black !important;
            background: white !important;
            box-sizing: border-box !important;
            width: 14mm !important;
            height: 14mm !important;
            border-radius: 0 !important; /* Optional: squares might print cleaner on thermal */
          }
          .no-print {
            display: none !important;
          }
        }
      `}</style>

      {/* Header */}
      <header className="flex justify-between items-center print:hidden">
        <div>
          <h2 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
            <Printer className="text-blue-600" />
            منابع و تولیدات آزمایشگاهی
          </h2>
          <p className="text-slate-500 text-sm mt-1">ساخت دیسک آنتی‌بیوتیک دست‌ساز و محاسبات دوز</p>
        </div>
        <div className="bg-blue-50 text-blue-700 px-3 py-1 rounded-full text-xs font-bold flex items-center gap-2">
           <Zap size={14} />
           بهینه‌شده برای چاپگر حرارتی (Thermal Printer)
        </div>
      </header>

      {/* Tabs */}
      <div className="flex gap-4 border-b border-slate-200 print:hidden">
        <button 
          onClick={() => setMode('designer')}
          className={`pb-3 px-4 text-sm font-medium transition-colors border-b-2 flex items-center gap-2 ${
            mode === 'designer' 
            ? 'border-blue-600 text-blue-600' 
            : 'border-transparent text-slate-500 hover:text-slate-700'
          }`}
        >
          <Grid size={16} />
          طراحی و چاپ دیسک
        </button>
        <button 
          onClick={() => setMode('calculator')}
          className={`pb-3 px-4 text-sm font-medium transition-colors border-b-2 flex items-center gap-2 ${
            mode === 'calculator' 
            ? 'border-blue-600 text-blue-600' 
            : 'border-transparent text-slate-500 hover:text-slate-700'
          }`}
        >
          <Calculator size={16} />
          محاسبه دوز محلول
        </button>
      </div>

      {mode === 'calculator' ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 print:hidden">
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-6">
              <h3 className="font-bold text-slate-700 border-b border-slate-100 pb-2">ورودی‌ها</h3>
              
              <div>
                <label className="block text-sm font-medium text-slate-600 mb-1">وزن قرص/کپسول منبع (mg)</label>
                <div className="relative">
                   <input 
                     type="number" 
                     value={calcSourceMg}
                     onChange={e => setCalcSourceMg(Number(e.target.value))}
                     className="w-full p-3 bg-slate-50 border border-slate-200 rounded-xl"
                   />
                   <span className="absolute left-3 top-3 text-slate-400 text-sm">میلی‌گرم</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-600 mb-1">قدرت دیسک هدف (mcg)</label>
                <div className="relative">
                   <input 
                     type="number" 
                     value={calcTargetMcg}
                     onChange={e => setCalcTargetMcg(Number(e.target.value))}
                     className="w-full p-3 bg-slate-50 border border-slate-200 rounded-xl"
                   />
                   <span className="absolute left-3 top-3 text-slate-400 text-sm">میکروگرم</span>
                </div>
                <p className="text-xs text-slate-400 mt-1">استاندارد معمول: ۱۰ یا ۳۰ میکروگرم</p>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-600 mb-1">حجم قطره پیپت شما (ul)</label>
                <div className="relative">
                   <input 
                     type="number" 
                     value={calcDropUl}
                     onChange={e => setCalcDropUl(Number(e.target.value))}
                     className="w-full p-3 bg-slate-50 border border-slate-200 rounded-xl"
                   />
                   <span className="absolute left-3 top-3 text-slate-400 text-sm">میکرولیتر</span>
                </div>
              </div>
            </div>
            
            {/* Results */}
            <div className="bg-blue-600 text-white p-6 rounded-2xl shadow-lg flex flex-col justify-center relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-white opacity-5 rounded-full -mr-10 -mt-10"></div>
              <h3 className="text-lg font-bold mb-6 flex items-center gap-2">
                <FlaskConical />
                دستورالعمل ساخت محلول
              </h3>
              
              <div className="space-y-6 relative z-10">
                <div className="bg-white/10 p-4 rounded-xl backdrop-blur-sm">
                  <span className="text-blue-100 text-xs uppercase block mb-1">غلظت مورد نیاز</span>
                  <span className="text-2xl font-black">{requiredConc.toFixed(2)} mg/ml</span>
                </div>

                <div className="bg-white text-blue-900 p-5 rounded-xl shadow-md">
                  <span className="text-blue-600 text-xs font-bold uppercase block mb-2">دستور نهایی</span>
                  <p className="font-medium leading-relaxed">
                    محتوای کپسول <span className="font-bold">{calcSourceMg} میلی‌گرمی</span> را در 
                    <span className="font-black text-xl mx-1 text-blue-700">{requiredSolvent.toFixed(1)}</span>
                    میلی‌لیتر آب مقطر استریل حل کنید.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* QC Voice Assistant */}
          <div className="space-y-4">
            <div className="bg-indigo-50 border border-indigo-100 p-4 rounded-2xl">
              <div className="flex items-start gap-3">
                 <div className="bg-indigo-100 p-2 rounded-lg text-indigo-600">
                   <Info size={24} />
                 </div>
                 <div>
                    <h4 className="font-bold text-indigo-900">مشاوره فنی و کنترل کیفیت</h4>
                    <p className="text-sm text-indigo-700 mt-1">
                      سوالات خود را در مورد نحوه استریل کردن، خشک کردن دیسک‌ها و شرایط نگهداری از کارشناس هوشمند بپرسید.
                    </p>
                 </div>
              </div>
            </div>
            <LiveVoiceAssistant initialContext={getCalculatorContext()} persona="qc" />
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[calc(100vh-200px)]">
          {/* Controls */}
          <div className="lg:col-span-4 bg-white p-5 rounded-2xl border border-slate-200 shadow-sm flex flex-col print:hidden">
            <h3 className="font-bold text-slate-700 mb-4">تنظیمات چاپ</h3>
            
            <div className="space-y-4 mb-6">
               {/* Shape */}
               <div>
                 <label className="text-xs font-bold text-slate-500 mb-2 block">الگوی برش</label>
                 <div className="flex gap-2">
                   <button 
                     onClick={() => setDiscShape('circle')}
                     className={`flex-1 py-2 rounded-lg text-sm font-medium border flex items-center justify-center gap-2 ${discShape === 'circle' ? 'bg-blue-50 border-blue-500 text-blue-700' : 'border-slate-200 text-slate-600'}`}
                   >
                     <Circle size={14} />
                     دایره (پانچ)
                   </button>
                   <button 
                     onClick={() => setDiscShape('square')}
                     className={`flex-1 py-2 rounded-lg text-sm font-medium border flex items-center justify-center gap-2 ${discShape === 'square' ? 'bg-blue-50 border-blue-500 text-blue-700' : 'border-slate-200 text-slate-600'}`}
                   >
                     <Scissors size={14} />
                     مربع (قیچی)
                   </button>
                 </div>
               </div>

               {/* Batch Quantity */}
               <div className="bg-slate-50 p-3 rounded-xl border border-slate-100">
                  <label className="text-xs font-bold text-slate-500 mb-2 block">تعداد در هر دسته (Batch Size)</label>
                  <div className="flex items-center gap-2">
                    <input 
                      type="number" 
                      min="1" 
                      max="1000"
                      value={batchSize}
                      onChange={(e) => setBatchSize(parseInt(e.target.value) || 0)}
                      className="w-20 p-2 text-center font-bold border border-slate-300 rounded-lg text-slate-700"
                    />
                    <div className="flex gap-1 flex-1">
                       <button onClick={() => setBatchSize(50)} className="flex-1 bg-white border hover:bg-slate-100 text-xs py-2 rounded-lg font-medium">50</button>
                       <button onClick={() => setBatchSize(100)} className="flex-1 bg-white border hover:bg-slate-100 text-xs py-2 rounded-lg font-medium">100</button>
                       <button onClick={() => setBatchSize(500)} className="flex-1 bg-white border hover:bg-slate-100 text-xs py-2 rounded-lg font-medium">500</button>
                    </div>
                  </div>
               </div>

               {/* Quick Add Buttons */}
               <div>
                 <label className="text-xs font-bold text-slate-500 mb-2 block">افزودن سریع (+{batchSize} عدد)</label>
                 <div className="flex flex-wrap gap-2">
                   {commonAntibiotics.map(ab => (
                     <button 
                       key={ab.code}
                       onClick={() => addToQueue(ab)}
                       className="px-3 py-2 bg-blue-50 hover:bg-blue-100 rounded-lg text-xs font-bold text-blue-700 border border-blue-100 transition-colors flex items-center gap-1"
                     >
                       <Layers size={14} />
                       {ab.code}
                     </button>
                   ))}
                 </div>
               </div>

               <div className="pt-4 border-t border-slate-100">
                 <label className="text-xs font-bold text-slate-500 mb-2 block">دیسک سفارشی (+{batchSize} عدد)</label>
                 <div className="flex gap-2">
                   <input 
                     placeholder="کد (مثلا AZM)" 
                     className="w-20 p-2 text-sm border border-slate-200 rounded-lg"
                     value={newDiscCode}
                     onChange={e => setNewDiscCode(e.target.value.toUpperCase())}
                     maxLength={3}
                   />
                   <button 
                     onClick={() => {
                        if(newDiscCode) {
                          addToQueue({name: 'Custom', code: newDiscCode});
                          setNewDiscCode('');
                        }
                     }}
                     className="bg-slate-800 text-white px-3 rounded-lg hover:bg-slate-700"
                   >
                     <Plus size={18} />
                   </button>
                 </div>
               </div>
            </div>

            <div className="mt-auto">
               <div className="bg-slate-50 p-3 rounded-xl mb-3 text-xs text-slate-500 flex gap-2">
                 <Info size={16} className="shrink-0" />
                 <span>دیسک‌ها برای پانچ استاندارد ۶ میلی‌متری طراحی شده‌اند. بعد از چاپ، با پانچ اداری سوراخ کنید.</span>
               </div>
               <button 
                 onClick={() => setShowPreviewModal(true)}
                 disabled={printQueue.length === 0}
                 className="w-full py-3 bg-blue-600 text-white rounded-xl font-bold flex items-center justify-center gap-2 hover:bg-blue-700 transition-colors shadow-lg shadow-blue-200"
               >
                 <Printer size={18} />
                 مشاهده پیش‌نمایش و چاپ
               </button>
               <button 
                 onClick={() => setPrintQueue([])}
                 className="w-full mt-2 py-2 text-red-500 text-sm hover:bg-red-50 rounded-lg"
               >
                 پاک کردن لیست
               </button>
            </div>
          </div>

          {/* Canvas Area (Visual representation) */}
          <div className="lg:col-span-8 bg-slate-100 rounded-2xl p-8 overflow-y-auto flex items-start justify-center border-2 border-dashed border-slate-200">
             <div className="w-[72mm] min-h-[500px] bg-white shadow-sm p-0 flex flex-wrap content-start gap-0 opacity-80 pointer-events-none grayscale">
                <div className="w-full text-center text-slate-400 my-4 font-bold text-[10px]">نمای رول (۸۰mm)</div>
                {printQueue.map((disc, idx) => (
                  <div key={idx} className="w-[14mm] h-[14mm] border border-slate-300 flex items-center justify-center text-[8px] box-border">{disc.code}</div>
                ))}
             </div>
          </div>
        </div>
      )}

      {/* PRINT PREVIEW MODAL */}
      {showPreviewModal && (
        <div id="preview-modal" className="fixed inset-0 z-50 bg-slate-900/95 flex flex-col animate-fade-in overflow-hidden">
           {/* Modal Header (No Print) */}
           <div className="no-print bg-slate-800 p-4 flex items-center justify-between border-b border-slate-700 text-white">
              <h3 className="font-bold text-lg flex items-center gap-2">
                <Printer className="text-blue-400" />
                پیش‌نمایش چاپ (Thermal 80mm)
              </h3>
              <div className="flex gap-3">
                <span className="bg-slate-700 px-3 py-1 rounded-full text-xs font-mono">
                   تعداد کل: {printQueue.length}
                </span>
                <button 
                  onClick={handlePrint}
                  className="bg-blue-600 hover:bg-blue-500 px-6 py-2 rounded-lg font-bold flex items-center gap-2"
                >
                  <Check size={18} />
                  تایید و پرینت
                </button>
                <button 
                  onClick={() => setShowPreviewModal(false)}
                  className="bg-slate-700 hover:bg-slate-600 px-4 py-2 rounded-lg"
                >
                  <X size={18} />
                </button>
              </div>
           </div>

           {/* Printable Content Scrollable Area */}
           <div className="flex-1 overflow-auto p-8 flex justify-center bg-slate-800">
              <div id="printable-area-wrapper">
                <div 
                   id="printable-area" 
                   className="bg-white p-0 shadow-2xl flex flex-wrap content-start gap-0"
                   style={{
                     width: '72mm',
                     minHeight: '100mm'
                   }}
                >
                   {printQueue.map((disc, idx) => (
                      <div 
                        key={idx}
                        className="disc-preview flex items-center justify-center font-black text-black border border-black relative box-border"
                        style={{
                          width: '14mm', 
                          height: '14mm', 
                          fontSize: '10px',
                          borderRadius: discShape === 'circle' ? '50%' : '0',
                        }}
                      >
                        {/* Crosshair helper for punching */}
                        <div className="absolute inset-0 flex items-center justify-center opacity-20 pointer-events-none">
                           <div className="w-full h-[1px] bg-black"></div>
                           <div className="h-full w-[1px] bg-black absolute"></div>
                        </div>
                        <span className="bg-white px-0.5 z-10">{disc.code}</span>
                      </div>
                   ))}
                </div>
              </div>
           </div>
        </div>
      )}
    </div>
  );
};
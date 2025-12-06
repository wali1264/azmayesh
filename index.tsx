
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
// ... [LabResourcesModule omitted - stays same] ...

// 3. Updated LiveLabModule using getLiveClient
const LiveLabModule = () => {
  const [active, setActive] = useState(false);
  const [micOn, setMicOn] = useState(true);
  const [cameraOn, setCameraOn] = useState(true);
  const [status, setStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
  const [volume, setVolume] = useState(0); 
  const [isMotionDetected, setIsMotionDetected] = useState(false); 

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const audioQueueRef = useRef<AudioBufferSourceNode[]>([]);
  const frameIntervalRef = useRef<number | null>(null);
  
  const prevFrameDataRef = useRef<Uint8ClampedArray | null>(null);
  const lastSendTimeRef = useRef<number>(0);
  const volumeRef = useRef<number>(0); 

  const detectMotion = (ctx: CanvasRenderingContext2D, width: number, height: number): boolean => {
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;
    let score = 0;
    
    if (prevFrameDataRef.current && prevFrameDataRef.current.length === data.length) {
       for (let i = 0; i < data.length; i += 32) { 
          const diff = Math.abs(data[i] - prevFrameDataRef.current[i]);
          if (diff > 30) score++;
       }
    }
    prevFrameDataRef.current = new Uint8ClampedArray(data);
    const threshold = (data.length / 32) * 0.02; 
    return score > threshold;
  };

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

  const startLocalStream = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: AUDIO_INPUT_SAMPLE_RATE,
          autoGainControl: true
        }
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      return stream;
    } catch (err) {
      console.error("Error accessing media devices:", err);
      setStatus('error');
      return null;
    }
  };

  const connectToGemini = async () => {
    initAudioContext();
    const stream = await startLocalStream();
    if (!stream) return;

    setStatus('connecting');
    setActive(true);

    try {
      const model = 'gemini-2.5-flash-native-audio-preview-09-2025'; 
      
      // Get a Random Client for Live Session to balance load
      const client = aiService.getLiveClient();

      const sessionPromise = client.live.connect({
        model: model,
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: `شما "دکتر اعظم" هستید.
          1. **شروع فوری:** به محض اتصال، سلام کن.
          2. **شخصیت:** همکار، شوخ‌طبع.`,
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } }
          }
        },
        callbacks: {
          onopen: () => {
            setStatus('connected');
            
            const ctx = audioContextRef.current!;
            const source = ctx.createMediaStreamSource(stream);
            // Use 4096 for buffer stability
            const processor = ctx.createScriptProcessor(4096, 1, 1);
            
            processor.onaudioprocess = (e) => {
              if (!micOn) return; 
              const inputData = e.inputBuffer.getChannelData(0);
              
              let sum = 0;
              for(let i=0; i<inputData.length; i+=100) sum += Math.abs(inputData[i]);
              const v = Math.min(100, (sum / (inputData.length/100)) * 500);
              setVolume(v);
              volumeRef.current = v;

              const pcmData = floatTo16BitPCM(inputData);
              const uint8 = new Uint8Array(pcmData.buffer);
              const base64 = arrayBufferToBase64(uint8.buffer);

              sessionPromise.then(session => {
                session.sendRealtimeInput({
                  media: {
                    mimeType: "audio/pcm;rate=16000",
                    data: base64
                  }
                });
              });
            };
            
            source.connect(processor);
            processor.connect(ctx.destination);
            processorRef.current = processor;
            sourceRef.current = source;

            frameIntervalRef.current = window.setInterval(() => {
              if (!cameraOn || !videoRef.current || !canvasRef.current) return;
              
              const video = videoRef.current;
              const canvas = canvasRef.current;
              const ctx2d = canvas.getContext('2d', { willReadFrequently: true });
              
              if (video.readyState === video.HAVE_ENOUGH_DATA && ctx2d) {
                const sendWidth = video.videoWidth / 3;
                const sendHeight = video.videoHeight / 3;
                
                canvas.width = sendWidth;
                canvas.height = sendHeight;
                ctx2d.drawImage(video, 0, 0, sendWidth, sendHeight);
                
                const hasMotion = detectMotion(ctx2d, sendWidth, sendHeight);
                setIsMotionDetected(hasMotion);

                const now = Date.now();
                const timeSinceLast = now - lastSendTimeRef.current;
                const isSpeaking = volumeRef.current > 10; 
                
                let shouldSend = false;

                if ((hasMotion || isSpeaking) && timeSinceLast > 1000) {
                   shouldSend = true;
                } else if (timeSinceLast > 3000) {
                   shouldSend = true;
                }

                if (shouldSend) {
                    const base64Data = canvas.toDataURL('image/jpeg', 0.6).split(',')[1];
                    sessionPromise.then(session => {
                      session.sendRealtimeInput({
                        media: {
                          mimeType: "image/jpeg",
                          data: base64Data
                        }
                      });
                    });
                    lastSendTimeRef.current = now;
                }
              }
            }, 200); 
          },
          onmessage: async (msg) => {
            if (msg.serverContent?.interrupted) {
              audioQueueRef.current.forEach(source => {
                try { source.stop(); } catch(e) {}
              });
              audioQueueRef.current = [];
              if (audioContextRef.current) {
                  nextStartTimeRef.current = audioContextRef.current.currentTime;
              }
              return;
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
            setStatus('disconnected');
            setActive(false);
          },
          onerror: (err) => {
            console.error("Session error:", err);
            setStatus('error');
            setActive(false);
          }
        }
      });

    } catch (e) {
      console.error(e);
      setStatus('error');
    }
  };

  const stopSession = () => {
    setActive(false);
    setStatus('disconnected');
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (processorRef.current && sourceRef.current) {
      processorRef.current.disconnect();
      sourceRef.current.disconnect();
    }
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
    }
    audioQueueRef.current.forEach(source => source.stop());
    audioQueueRef.current = [];
    nextStartTimeRef.current = 0;
  };

  useEffect(() => {
    return () => stopSession();
  }, []);

  return (
    <div className="flex flex-col h-[calc(100vh-140px)] md:h-[calc(100vh-80px)] space-y-4">
      <header className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
            <Microscope className="text-blue-600" />
            آزمایشگاه زنده
          </h2>
          <p className="text-slate-500 text-sm mt-1">نظارت هوشمند و دستیار صوتی لحظه‌ای (پشتیبانی از مولتی-کلید)</p>
        </div>
        
        {status === 'connected' && (
          <div className="flex items-center gap-3">
             <div className="flex items-center gap-1.5 px-3 py-1 bg-slate-100 rounded-full border border-slate-200">
               <Eye size={14} className={isMotionDetected ? "text-green-600" : "text-yellow-500"} />
               <span className="text-[10px] font-bold text-slate-600">
                 {isMotionDetected ? "ارسال فعال" : "صرفه‌جویی"}
               </span>
               <div className={`w-2 h-2 rounded-full ${isMotionDetected ? "bg-green-500 animate-pulse" : "bg-yellow-400"}`}></div>
             </div>

             <div className="flex items-center gap-2 bg-green-100 text-green-700 px-3 py-1.5 rounded-full animate-pulse">
               <div className="w-2 h-2 bg-green-600 rounded-full"></div>
               <span className="text-xs font-bold">اتصال ایمن</span>
             </div>
          </div>
        )}
      </header>

      <div className="flex-1 bg-black rounded-2xl overflow-hidden relative shadow-xl border border-slate-300 group">
        <video 
          ref={videoRef} 
          autoPlay 
          playsInline 
          muted 
          className={`w-full h-full object-cover transition-opacity duration-500 ${cameraOn ? 'opacity-100' : 'opacity-0'}`}
        />
        <canvas ref={canvasRef} className="hidden" />
        
        {!active && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900 text-white z-10">
            <div className="w-24 h-24 rounded-full bg-slate-800 flex items-center justify-center mb-6 border border-slate-700">
              <Camera size={40} className="text-slate-500" />
            </div>
            <h3 className="text-xl font-bold mb-2">دوربین غیرفعال است</h3>
            <p className="text-slate-400 text-sm max-w-md text-center px-4 mb-8">
              برای شروع نظارت هوشمند و دریافت مشاوره صوتی، ارتباط با دستیار را برقرار کنید.
            </p>
            <button 
              onClick={connectToGemini}
              className="bg-blue-600 hover:bg-blue-500 text-white px-8 py-3 rounded-xl font-bold transition-all transform hover:scale-105 shadow-lg shadow-blue-900/50 flex items-center gap-2"
            >
              <Power size={20} />
              شروع دستیار هوشمند
            </button>
          </div>
        )}

        {status === 'connecting' && (
          <div className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center z-20 backdrop-blur-sm">
            <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
            <span className="text-white font-medium">در حال انتخاب بهترین سرور...</span>
          </div>
        )}

        {status === 'error' && (
          <div className="absolute inset-0 bg-red-900/90 flex flex-col items-center justify-center z-20">
            <AlertCircle size={48} className="text-red-200 mb-4" />
            <h3 className="text-xl font-bold text-white mb-2">خطا در اتصال</h3>
            <p className="text-red-200 text-sm mb-6">لطفاً صفحه را رفرش کنید یا دسترسی‌ها را چک کنید</p>
            <button 
              onClick={() => { setStatus('disconnected'); setActive(false); }}
              className="bg-white text-red-900 px-6 py-2 rounded-lg font-bold hover:bg-red-50"
            >
              بازگشت
            </button>
          </div>
        )}

        {active && (
          <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-black/90 to-transparent flex items-end justify-between z-10">
            
            <div className="flex items-center gap-4">
               <div className="flex flex-col">
                 <div className="flex items-center gap-2 mb-1">
                   <div className={`w-2 h-2 rounded-full ${micOn ? 'bg-blue-500' : 'bg-red-500'}`}></div>
                   <span className="text-xs text-slate-300 font-mono">MIC INPUT</span>
                 </div>
                 <div className="flex items-end gap-0.5 h-8">
                    {[...Array(5)].map((_, i) => (
                      <div 
                        key={i} 
                        className={`w-1.5 rounded-sm transition-all duration-100 ${volume > i * 20 ? 'bg-blue-500' : 'bg-slate-700'}`}
                        style={{ height: volume > i * 20 ? `${Math.max(20, Math.random() * 100)}%` : '20%' }}
                      ></div>
                    ))}
                 </div>
               </div>
            </div>

            <div className="flex items-center gap-3">
              <button 
                onClick={() => setMicOn(!micOn)}
                disabled={status !== 'connected'}
                className={`p-4 rounded-full backdrop-blur-md transition-all ${
                  micOn ? 'bg-slate-800/50 text-white hover:bg-slate-700/50' : 'bg-red-500/80 text-white hover:bg-red-600/80'
                }`}
              >
                {micOn ? <Mic size={24} /> : <MicOff size={24} />}
              </button>
              
              <button 
                onClick={() => setCameraOn(!cameraOn)}
                disabled={status !== 'connected'}
                className={`p-4 rounded-full backdrop-blur-md transition-all ${
                  cameraOn ? 'bg-slate-800/50 text-white hover:bg-slate-700/50' : 'bg-red-500/80 text-white hover:bg-red-600/80'
                }`}
              >
                {cameraOn ? <Video size={24} /> : <VideoOff size={24} />}
              </button>

              <button 
                onClick={stopSession}
                className="p-4 rounded-full bg-red-600 text-white hover:bg-red-700 transition-all shadow-lg shadow-red-900/50"
              >
                <Power size={24} />
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex items-start gap-3">
         <div className="bg-blue-100 p-2 rounded-lg text-blue-600">
           <Zap size={20} />
         </div>
         <div>
           <h4 className="font-bold text-slate-800 text-sm">Load Balancing فعال است</h4>
           <p className="text-xs text-slate-500 mt-1">
             برای جلوگیری از قطعی، هر جلسه جدید از یک سرور متفاوت (کلید تصادفی) استفاده می‌کند.
             <br/>
             در صورت بروز خطا در یک سرور، سیستم به صورت خودکار به سرور بعدی سوئیچ می‌کند.
           </p>
         </div>
      </div>
    </div>
  );
};

const ClinicalConsoleModule = () => {
  // State for Inputs
  const [vitals, setVitals] = useState<VitalsData>({
    temp: { value: 37.0, enabled: true },
    hr: { value: 72, enabled: true },
    bp: { sys: 120, dia: 80, enabled: true },
    spo2: { value: 98, enabled: true },
    weight: { value: 70, enabled: true }
  });
  const [historyText, setHistoryText] = useState('');
  const [historyLang, setHistoryLang] = useState<'fa' | 'en'>('fa');
  const [isListeningHistory, setIsListeningHistory] = useState(false);
  const [uploadedImages, setUploadedImages] = useState<string[]>([]);
  
  // State for Processing
  const [isDiagnosing, setIsDiagnosing] = useState(false);
  const [diagnosisResult, setDiagnosisResult] = useState<ClinicalDiagnosisResult | null>(null);

  // Vitals Helpers
  const toggleVital = (key: keyof VitalsData) => {
    setVitals(prev => ({
      ...prev,
      [key]: { ...prev[key], enabled: !prev[key].enabled }
    }));
  };

  const updateSimpleVital = (key: keyof VitalsData, val: number) => {
    // For simple values (temp, hr, spo2, weight)
    setVitals(prev => ({
      ...prev,
      [key]: { ...prev[key], value: val }
    }));
  };
  
  const updateBP = (type: 'sys' | 'dia', val: number) => {
    setVitals(prev => ({
      ...prev,
      bp: { ...prev.bp, [type]: val }
    }));
  };

  // History Speech Helper
  const toggleHistorySpeech = () => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert('مرورگر پشتیبانی نمی‌کند');
      return;
    }

    if (isListeningHistory) {
      setIsListeningHistory(false);
      return;
    }

    setIsListeningHistory(true);
    const recognition = new SpeechRecognition();
    recognition.lang = historyLang === 'fa' ? 'fa-IR' : 'en-US';
    recognition.continuous = false;
    
    recognition.onresult = (e: any) => {
      const text = e.results[0][0].transcript;
      setHistoryText(prev => prev + (prev ? ' ' : '') + text);
      setIsListeningHistory(false);
    };
    recognition.onerror = () => setIsListeningHistory(false);
    recognition.onend = () => setIsListeningHistory(false);
    recognition.start();
  };

  // Image Upload Helper
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setUploadedImages(prev => [...prev, (reader.result as string).split(',')[1]]);
      };
      reader.readAsDataURL(file);
    }
  };

  // Main Action
  const handleDiagnosis = async () => {
    setIsDiagnosing(true);
    const result = await aiService.clinicalDiagnosis(vitals, historyText, uploadedImages);
    setDiagnosisResult(result);
    setIsDiagnosing(false);
  };

  // Prepare Context for Voice Assistant
  const getContextString = () => {
    let ctx = `Patient Vitals: `;
    if(vitals.temp.enabled) ctx += `Temp ${vitals.temp.value}C, `;
    if(vitals.hr.enabled) ctx += `HR ${vitals.hr.value}, `;
    if(vitals.bp.enabled) ctx += `BP ${vitals.bp.sys}/${vitals.bp.dia}, `;
    
    ctx += `\nClinical History: ${historyText || 'N/A'}`;
    if (diagnosisResult) {
      ctx += `\nPrevious AI Diagnosis: ${diagnosisResult.diagnosis}`;
      ctx += `\nSuggested Media: ${diagnosisResult.media_recommendation.standard.name}`;
    }
    return ctx;
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <header className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
            <Stethoscope className="text-blue-600" />
            کنسول تشخیص بالینی و کشت
          </h2>
          <p className="text-slate-500 text-sm mt-1">اتاق فرمان پزشکی برای تشخیص هوشمند و انتخاب محیط کشت</p>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* LEFT COLUMN: Inputs (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          
          {/* CARD 1: VITALS (REDESIGNED) */}
          <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm relative overflow-hidden">
            <div className="flex items-center gap-2 mb-4 text-slate-700 font-bold border-b border-slate-100 pb-2">
              <Activity size={18} />
              <h3>تنظیم علائم حیاتی بیمار</h3>
              <span className="text-xs font-normal text-slate-400 mr-auto">برای فعال‌سازی تیک بزنید</span>
            </div>
            
            {/* BP Card - Full Width */}
            <div className={`mb-6 p-4 rounded-xl border-2 transition-all ${vitals.bp.enabled ? 'border-blue-100 bg-blue-50/30' : 'border-slate-100 bg-slate-50 opacity-60'}`}>
               <div className="flex justify-between items-center mb-4">
                  <div className="flex items-center gap-2">
                    <input 
                      type="checkbox" 
                      checked={vitals.bp.enabled} 
                      onChange={() => toggleVital('bp')}
                      className="w-5 h-5 rounded text-blue-600 focus:ring-blue-500 cursor-pointer"
                    />
                    <label className="text-base font-bold text-slate-700 flex items-center gap-2 cursor-pointer" onClick={() => toggleVital('bp')}>
                       فشار خون (BP)
                    </label>
                  </div>
                  <div className="text-2xl font-black text-blue-900 tracking-tight">
                    {vitals.bp.sys} <span className="text-slate-400 text-lg">/</span> {vitals.bp.dia}
                  </div>
               </div>
               
               <div className="grid grid-cols-1 gap-6">
                 {/* Systolic */}
                 <div className="relative pt-1">
                   <div className="flex justify-between text-xs font-bold text-slate-400 mb-1">
                     <span>SYS (سیستولیک)</span>
                     <span>{vitals.bp.sys}</span>
                   </div>
                   <input 
                    type="range" min="70" max="220" step="1"
                    value={vitals.bp.sys}
                    onChange={(e) => updateBP('sys', parseInt(e.target.value))}
                    disabled={!vitals.bp.enabled}
                    className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                   />
                 </div>
                 {/* Diastolic */}
                 <div className="relative pt-1">
                   <div className="flex justify-between text-xs font-bold text-slate-400 mb-1">
                     <span>DIA (دیاستولیک)</span>
                     <span>{vitals.bp.dia}</span>
                   </div>
                   <input 
                    type="range" min="40" max="130" step="1"
                    value={vitals.bp.dia}
                    onChange={(e) => updateBP('dia', parseInt(e.target.value))}
                    disabled={!vitals.bp.enabled}
                    className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-sky-500"
                   />
                 </div>
               </div>
            </div>

            {/* Other Vitals Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              
              {/* SpO2 */}
              <div className={`p-4 rounded-xl border border-slate-100 transition-all ${vitals.spo2.enabled ? 'bg-cyan-50/50' : 'bg-slate-50 opacity-60'}`}>
                <div className="flex justify-between items-center mb-3">
                  <div className="flex items-center gap-2">
                    <input type="checkbox" checked={vitals.spo2.enabled} onChange={() => toggleVital('spo2')} className="w-4 h-4 rounded text-cyan-600" />
                    <label className="font-bold text-sm flex items-center gap-1"><Wind size={16} className="text-cyan-600"/> اکسیژن (SPO2)</label>
                  </div>
                  <span className={`font-black text-lg ${vitals.spo2.value < 95 ? 'text-red-500' : 'text-cyan-700'}`}>{vitals.spo2.value}%</span>
                </div>
                <input type="range" min="70" max="100" value={vitals.spo2.value} onChange={(e) => updateSimpleVital('spo2', parseInt(e.target.value))} disabled={!vitals.spo2.enabled} className="w-full h-2 bg-slate-200 rounded-lg accent-cyan-600" />
              </div>

              {/* Heart Rate */}
              <div className={`p-4 rounded-xl border border-slate-100 transition-all ${vitals.hr.enabled ? 'bg-rose-50/50' : 'bg-slate-50 opacity-60'}`}>
                <div className="flex justify-between items-center mb-3">
                  <div className="flex items-center gap-2">
                    <input type="checkbox" checked={vitals.hr.enabled} onChange={() => toggleVital('hr')} className="w-4 h-4 rounded text-rose-600" />
                    <label className="font-bold text-sm flex items-center gap-1"><Heart size={16} className="text-rose-600"/> ضربان (HR)</label>
                  </div>
                  <span className={`font-black text-lg ${vitals.hr.value > 100 ? 'text-rose-600' : 'text-slate-700'}`}>{vitals.hr.value}</span>
                </div>
                <input type="range" min="40" max="200" value={vitals.hr.value} onChange={(e) => updateSimpleVital('hr', parseInt(e.target.value))} disabled={!vitals.hr.enabled} className="w-full h-2 bg-slate-200 rounded-lg accent-rose-500" />
              </div>

              {/* Weight */}
              <div className={`p-4 rounded-xl border border-slate-100 transition-all ${vitals.weight.enabled ? 'bg-indigo-50/50' : 'bg-slate-50 opacity-60'}`}>
                <div className="flex justify-between items-center mb-3">
                  <div className="flex items-center gap-2">
                    <input type="checkbox" checked={vitals.weight.enabled} onChange={() => toggleVital('weight')} className="w-4 h-4 rounded text-indigo-600" />
                    <label className="font-bold text-sm flex items-center gap-1"><Scale size={16} className="text-indigo-600"/> وزن (Weight)</label>
                  </div>
                  <span className="font-black text-lg text-indigo-900">{vitals.weight.value} <span className="text-xs font-medium text-slate-400">kg</span></span>
                </div>
                <input type="range" min="3" max="150" value={vitals.weight.value} onChange={(e) => updateSimpleVital('weight', parseInt(e.target.value))} disabled={!vitals.weight.enabled} className="w-full h-2 bg-slate-200 rounded-lg accent-indigo-600" />
              </div>

              {/* Temperature */}
              <div className={`p-4 rounded-xl border border-slate-100 transition-all ${vitals.temp.enabled ? 'bg-orange-50/50' : 'bg-slate-50 opacity-60'}`}>
                <div className="flex justify-between items-center mb-3">
                  <div className="flex items-center gap-2">
                    <input type="checkbox" checked={vitals.temp.enabled} onChange={() => toggleVital('temp')} className="w-4 h-4 rounded text-orange-600" />
                    <label className="font-bold text-sm flex items-center gap-1"><Thermometer size={16} className="text-orange-600"/> دما (Temp)</label>
                  </div>
                  <span className={`font-black text-lg ${vitals.temp.value > 37.5 ? 'text-orange-600' : 'text-slate-700'}`}>{vitals.temp.value}°c</span>
                </div>
                <input type="range" min="35" max="42" step="0.1" value={vitals.temp.value} onChange={(e) => updateSimpleVital('temp', parseFloat(e.target.value))} disabled={!vitals.temp.enabled} className="w-full h-2 bg-slate-200 rounded-lg accent-orange-500" />
              </div>

            </div>
          </div>

          {/* CARD 2: CLINICAL HISTORY */}
          <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
            <div className="flex items-center justify-between mb-3 border-b border-slate-100 pb-2">
              <div className="flex items-center gap-2 text-slate-700 font-bold">
                <FileText size={18} />
                <h3>شرح حال و علائم بالینی</h3>
              </div>
              <div className="flex items-center gap-2">
                 <button 
                   onClick={() => setHistoryLang(l => l === 'fa' ? 'en' : 'fa')}
                   className="text-xs font-mono bg-slate-100 px-2 py-1 rounded flex items-center gap-1 hover:bg-slate-200"
                 >
                   <Languages size={12} />
                   {historyLang === 'fa' ? 'FA' : 'EN'}
                 </button>
                 <button 
                   onClick={toggleHistorySpeech}
                   className={`p-2 rounded-lg transition-all ${isListeningHistory ? 'bg-red-500 text-white animate-pulse shadow-md' : 'bg-blue-50 text-blue-600 hover:bg-blue-100'}`}
                   title="شروع دیکته صوتی"
                 >
                   {isListeningHistory ? <MicOff size={16} /> : <Mic size={16} />}
                 </button>
              </div>
            </div>
            <textarea 
              value={historyText}
              onChange={(e) => setHistoryText(e.target.value)}
              placeholder={historyLang === 'fa' ? "شرح حال بیمار را تایپ کنید یا بگویید (مثلا: تب بالا، گلودرد شدید، مراجعه بعد از سفر...)" : "Dictate patient history..."}
              className="w-full h-32 p-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:outline-none text-sm resize-none"
              dir={historyLang === 'fa' ? 'rtl' : 'ltr'}
            />
          </div>

          {/* CARD 3: ATTACHMENTS */}
          <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
            <div className="flex items-center gap-2 text-slate-700 font-bold mb-3 border-b border-slate-100 pb-2">
              <FilePlus size={18} />
              <h3>مدارک پزشکی (نسخه، آزمایش خون، ...)</h3>
            </div>
            
            <div className="grid grid-cols-4 gap-4">
              {uploadedImages.map((img, idx) => (
                <div key={idx} className="relative aspect-square rounded-xl overflow-hidden border border-slate-200 group">
                  <img src={`data:image/jpeg;base64,${img}`} className="w-full h-full object-cover" />
                  <button 
                    onClick={() => setUploadedImages(prev => prev.filter((_, i) => i !== idx))}
                    className="absolute top-1 right-1 bg-red-500 text-white p-1 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <X size={12} />
                  </button>
                </div>
              ))}
              
              <label className="aspect-square border-2 border-dashed border-slate-300 rounded-xl flex flex-col items-center justify-center cursor-pointer hover:bg-slate-50 hover:border-blue-400 transition-colors text-slate-400 hover:text-blue-500">
                <Upload size={24} className="mb-1" />
                <span className="text-[10px] font-bold">آپلود عکس</span>
                <input type="file" accept="image/*" className="hidden" onChange={handleImageUpload} />
              </label>
            </div>
          </div>

          {/* MAIN ACTION BUTTON */}
          <button 
            onClick={handleDiagnosis}
            disabled={isDiagnosing}
            className={`w-full py-4 rounded-2xl font-black text-lg shadow-xl shadow-blue-200 flex items-center justify-center gap-3 transition-all transform hover:scale-[1.01] ${
              isDiagnosing 
                ? 'bg-slate-100 text-slate-400 cursor-wait' 
                : 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white hover:from-blue-700 hover:to-indigo-700'
            }`}
          >
            {isDiagnosing ? (
              <>
                <div className="w-5 h-5 border-2 border-slate-400 border-t-transparent rounded-full animate-spin"></div>
                در حال مشاوره با دکتر اعظم...
              </>
            ) : (
              <>
                <Brain size={24} />
                تشخیص هوشمند و پیشنهاد محیط کشت
              </>
            )}
          </button>
        </div>

        {/* RIGHT COLUMN: Results & AI (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          
          {/* AI RESULT CARD */}
          {diagnosisResult ? (
            <div className="space-y-4 animate-slide-up">
              {/* Diagnosis Header */}
              <div className="bg-gradient-to-br from-indigo-900 to-slate-900 text-white p-5 rounded-2xl shadow-lg relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-400 to-purple-500"></div>
                <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">تشخیص دکتر اعظم</h4>
                <h2 className="text-xl font-bold mb-2">{diagnosisResult.diagnosis}</h2>
                <p className="text-sm text-slate-300 leading-relaxed opacity-90">{diagnosisResult.reasoning}</p>
              </div>

              {/* Prescription */}
              <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-sm">
                <div className="flex items-center gap-2 text-slate-700 font-bold mb-2">
                  <Pill size={18} className="text-emerald-600" />
                  <h3>پیشنهاد درمانی (Empiric)</h3>
                </div>
                <div className="bg-emerald-50 text-emerald-900 p-3 rounded-xl text-sm font-medium border border-emerald-100">
                  {diagnosisResult.prescription_suggestion}
                </div>
              </div>

              {/* Media Recommendation */}
              <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm ring-2 ring-blue-100">
                <div className="flex items-center gap-2 text-slate-700 font-bold mb-4">
                  <FlaskConical size={18} className="text-blue-600" />
                  <h3>محیط کشت پیشنهادی</h3>
                </div>

                {/* Standard Option */}
                <div className="mb-4">
                  <span className="text-xs font-bold text-slate-400 uppercase block mb-1">گزینه استاندارد (Gold Standard)</span>
                  <div className="bg-blue-50 border border-blue-200 rounded-xl p-3">
                    <h4 className="font-bold text-blue-800">{diagnosisResult.media_recommendation.standard.name}</h4>
                    <p className="text-xs text-blue-600 mt-1">{diagnosisResult.media_recommendation.standard.description}</p>
                  </div>
                </div>

                {/* Emergency Option */}
                <div>
                  <span className="text-xs font-bold text-slate-400 uppercase block mb-1">گزینه اضطراری (Emergency/Handmade)</span>
                  <div className="bg-orange-50 border border-orange-200 rounded-xl p-3">
                    <h4 className="font-bold text-orange-800">{diagnosisResult.media_recommendation.emergency.name}</h4>
                    <p className="text-xs text-orange-700 mt-1 font-mono leading-relaxed">
                      دستورالعمل: {diagnosisResult.media_recommendation.emergency.recipe}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            // Placeholder State
            <div className="h-64 bg-slate-100 rounded-2xl border-2 border-dashed border-slate-300 flex flex-col items-center justify-center text-slate-400 p-8 text-center">
              <Brain size={48} className="mb-4 opacity-20" />
              <p className="font-medium">منتظر داده‌های بیمار...</p>
              <p className="text-xs mt-2 opacity-70">علائم حیاتی و شرح حال را وارد کنید تا هوش مصنوعی پردازش را آغاز کند.</p>
            </div>
          )}

          {/* Quick Chat for Consultation */}
          <div className="mt-4">
            <LiveVoiceAssistant initialContext={getContextString()} persona="clinical" />
          </div>
        </div>
      </div>
    </div>
  );
};

const AnalysisModule = ({ onSave }: { onSave: (result: AnalysisResult) => void }) => {
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [saved, setSaved] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
        setResult(null);
        setSaved(false);
      };
      reader.readAsDataURL(file);
    }
  };

  const startAnalysis = async () => {
    if (!imagePreview) return;
    setIsAnalyzing(true);
    const base64Data = imagePreview.split(',')[1];
    
    const analysisResult = await aiService.analyzePlateImage(base64Data);
    if (analysisResult) {
      setResult(analysisResult);
    }
    setIsAnalyzing(false);
  };

  const handleSave = () => {
    if (result) {
      onSave(result);
      setSaved(true);
    }
  };

  const resetAnalysis = () => {
    setImagePreview(null);
    setResult(null);
    setSaved(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="space-y-6">
      <header className="flex justify-between items-end">
        <div>
          <h2 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
            <Activity className="text-blue-600" />
            آنالیز هوشمند و تشخیص
          </h2>
          <p className="text-slate-500 text-sm mt-1">اسکن پلیت، تشخیص زودهنگام و آنتی‌بیوگرام دیجیتال</p>
        </div>
        <div className="flex gap-2">
           {result && !saved && (
             <button onClick={handleSave} className="px-4 py-2 bg-green-600 text-white rounded-xl text-sm font-bold shadow-lg shadow-green-200 hover:bg-green-700 transition-colors flex items-center gap-2">
               <Save size={16} />
               ذخیره در پرونده
             </button>
           )}
           {result && (
             <button onClick={resetAnalysis} className="text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1 bg-white px-3 py-2 rounded-xl border border-blue-100">
               <RefreshCw size={14} />
               آنالیز جدید
             </button>
           )}
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[calc(100vh-200px)]">
        <div className="bg-white rounded-2xl p-4 border border-slate-200 shadow-sm flex flex-col relative overflow-hidden">
          {!imagePreview ? (
            <div 
              className="flex-1 border-2 border-dashed border-slate-300 rounded-xl flex flex-col items-center justify-center bg-slate-50/50 hover:bg-slate-50 transition-colors cursor-pointer group"
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="w-16 h-16 bg-blue-100 text-blue-500 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                <Upload size={32} />
              </div>
              <h3 className="font-bold text-slate-700">آپلود تصویر پلیت</h3>
              <p className="text-slate-400 text-xs mt-2 text-center max-w-xs">
                برای شروع آنالیز، عکس پلیت کشت را اینجا رها کنید یا کلیک کنید.
              </p>
              <input 
                ref={fileInputRef}
                type="file" 
                accept="image/*" 
                className="hidden" 
                onChange={handleFileSelect}
              />
              <button className="mt-6 px-4 py-2 bg-white border border-slate-200 text-slate-600 rounded-lg text-sm font-medium shadow-sm flex items-center gap-2">
                <Camera size={16} />
                استفاده از دوربین
              </button>
            </div>
          ) : (
            <>
              <div className="flex-1 relative rounded-xl overflow-hidden bg-black flex items-center justify-center">
                 <img src={imagePreview} alt="Petri Dish" className="max-w-full max-h-full object-contain" />
                 {isAnalyzing && (
                   <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center backdrop-blur-sm">
                     <div className="w-12 h-12 border-4 border-white border-t-transparent rounded-full animate-spin mb-4"></div>
                     <span className="text-white font-medium animate-pulse">در حال آنالیز میکروبی...</span>
                   </div>
                 )}
              </div>
              {!result && !isAnalyzing && (
                <div className="mt-4 flex gap-3">
                  <button 
                    onClick={startAnalysis}
                    className="flex-1 bg-blue-600 text-white py-3 rounded-xl font-bold hover:bg-blue-700 transition-colors shadow-lg shadow-blue-200 flex items-center justify-center gap-2"
                  >
                    <Activity size={20} />
                    شروع پردازش هوشمند
                  </button>
                  <button 
                    onClick={resetAnalysis}
                    className="px-4 bg-slate-100 text-slate-600 rounded-xl hover:bg-slate-200"
                  >
                    <X size={20} />
                  </button>
                </div>
              )}
            </>
          )}
        </div>

        <div className="flex flex-col h-full overflow-y-auto space-y-4">
          {!result ? (
            <div className="bg-slate-50 border border-slate-200 rounded-2xl h-full flex flex-col items-center justify-center text-center p-8">
              <div className="w-20 h-20 bg-white rounded-full shadow-sm flex items-center justify-center mb-4 text-slate-300">
                <FileText size={40} />
              </div>
              <h3 className="text-lg font-bold text-slate-700 mb-2">منتظر داده‌های آنالیز</h3>
              <p className="text-slate-500 text-sm max-w-sm">
                پس از آپلود تصویر، هوش مصنوعی کلونی‌ها را شمارش کرده و نتایج آنتی‌بیوگرام را تفسیر می‌کند.
              </p>
            </div>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-sm">
                   <h4 className="text-xs font-bold text-slate-400 mb-2 uppercase tracking-wider">تشخیص احتمالی</h4>
                   <div className="flex items-start gap-3">
                     <div className="bg-indigo-100 p-2 rounded-lg text-indigo-600">
                       <Microscope size={24} />
                     </div>
                     <div>
                       <div className="font-bold text-slate-800 text-lg leading-tight">{result.organism_suspicion}</div>
                       <div className="text-xs text-indigo-600 font-medium mt-1 bg-indigo-50 inline-block px-2 py-0.5 rounded-full">
                         اطمینان: {result.confidence}
                       </div>
                     </div>
                   </div>
                </div>

                <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-sm">
                   <h4 className="text-xs font-bold text-slate-400 mb-2 uppercase tracking-wider">مرحله رشد</h4>
                   <div className="flex items-start gap-3">
                     <div className="bg-orange-100 p-2 rounded-lg text-orange-600">
                       <Clock size={24} />
                     </div>
                     <div>
                       <div className="font-bold text-slate-800 text-lg leading-tight">{result.growth_stage}</div>
                       <div className="text-xs text-slate-500 mt-1 line-clamp-1">{result.colony_morphology}</div>
                     </div>
                   </div>
                </div>
              </div>

              <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden flex-1">
                <div className="px-5 py-4 border-b border-slate-100 flex justify-between items-center">
                  <h4 className="font-bold text-slate-800 flex items-center gap-2">
                    <CheckCircle size={18} className="text-green-500" />
                    نتایج آنتی‌بیوگرام (AST)
                  </h4>
                  <span className="text-[10px] text-slate-400">استاندارد CLSI</span>
                </div>
                
                {result.antibiotic_results.length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="w-full text-right text-sm">
                      <thead className="bg-slate-50 text-slate-500 font-medium">
                        <tr>
                          <th className="px-5 py-3 text-right">آنتی‌بیوتیک</th>
                          <th className="px-5 py-3 text-center">قطر هاله (mm)</th>
                          <th className="px-5 py-3 text-center">تفسیر</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {result.antibiotic_results.map((ab, idx) => (
                          <tr key={idx} className="hover:bg-slate-50/50">
                            <td className="px-5 py-3 font-medium text-slate-700">{ab.name}</td>
                            <td className="px-5 py-3 text-center font-mono text-slate-500">{ab.zone_size_mm}</td>
                            <td className="px-5 py-3 text-center">
                              <span className={`px-2 py-1 rounded-full text-xs font-bold ${
                                ab.interpretation === 'Sensitive' ? 'bg-green-100 text-green-700' :
                                ab.interpretation === 'Resistant' ? 'bg-red-100 text-red-700' :
                                'bg-yellow-100 text-yellow-700'
                              }`}>
                                {ab.interpretation === 'Sensitive' ? 'حساس (S)' : 
                                 ab.interpretation === 'Resistant' ? 'مقاوم (R)' : 'بینابینی (I)'}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="p-8 text-center text-slate-400">
                    <p>هیچ دیسک آنتی‌بیوتیکی در تصویر شناسایی نشد.</p>
                  </div>
                )}
              </div>

              <div className="bg-blue-50 border border-blue-100 rounded-2xl p-4">
                <h4 className="font-bold text-blue-800 text-sm mb-2 flex items-center gap-2">
                  <Info size={16} />
                  پیشنهاد بالینی هوشمند
                </h4>
                <p className="text-sm text-blue-700 leading-relaxed text-justify">
                  {result.recommendation}
                </p>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

const SettingsModule = ({ 
  history, 
  onClearHistory, 
  profile, 
  setProfile 
}: { 
  history: AnalysisResult[], 
  onClearHistory: () => void,
  profile: UserProfile,
  setProfile: (p: UserProfile) => void
}) => {
  const [activeTab, setActiveTab] = useState<'profile' | 'history'>('history');

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
          <Settings className="text-slate-600" />
          تنظیمات و پرونده‌ها
        </h2>
        <p className="text-slate-500 text-sm mt-1">مدیریت پروفایل کاربری و مشاهده سوابق آزمایشات</p>
      </header>

      <div className="flex gap-4 border-b border-slate-200">
        <button 
          onClick={() => setActiveTab('history')}
          className={`pb-3 px-4 text-sm font-medium transition-colors border-b-2 ${
            activeTab === 'history' 
            ? 'border-blue-600 text-blue-600' 
            : 'border-transparent text-slate-500 hover:text-slate-700'
          }`}
        >
          تاریخچه آزمایشات ({history.length})
        </button>
        <button 
          onClick={() => setActiveTab('profile')}
          className={`pb-3 px-4 text-sm font-medium transition-colors border-b-2 ${
            activeTab === 'profile' 
            ? 'border-blue-600 text-blue-600' 
            : 'border-transparent text-slate-500 hover:text-slate-700'
          }`}
        >
          پروفایل و سیستم
        </button>
      </div>

      <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm min-h-[400px]">
        {activeTab === 'history' ? (
          <div className="space-y-4">
             <div className="flex justify-between items-center mb-4">
               <h3 className="font-bold text-slate-700">آخرین آزمایش‌های انجام شده</h3>
               {history.length > 0 && (
                 <button 
                   onClick={onClearHistory}
                   className="text-xs text-red-500 hover:bg-red-50 px-3 py-1.5 rounded-lg flex items-center gap-1 transition-colors"
                 >
                   <Trash2 size={14} />
                   پاکسازی تاریخچه
                 </button>
               )}
             </div>
             
             {history.length === 0 ? (
               <div className="text-center py-12 text-slate-400">
                 <History size={48} className="mx-auto mb-3 opacity-20" />
                 <p>هنوز هیچ آزمایشی ثبت نشده است.</p>
               </div>
             ) : (
               <div className="space-y-3">
                 {history.slice().reverse().map((item) => (
                   <div key={item.id} className="flex items-center justify-between p-4 border border-slate-100 rounded-xl hover:bg-slate-50 transition-colors">
                     <div className="flex items-center gap-4">
                        <div className="w-10 h-10 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center">
                          <Microscope size={20} />
                        </div>
                        <div>
                          <div className="font-bold text-slate-800">{item.organism_suspicion}</div>
                          <div className="text-xs text-slate-500">
                            {new Date(item.timestamp).toLocaleDateString('fa-IR')} • {new Date(item.timestamp).toLocaleTimeString('fa-IR', {hour: '2-digit', minute:'2-digit'})}
                          </div>
                        </div>
                     </div>
                     <div className="flex items-center gap-3">
                       <span className="text-xs bg-slate-100 text-slate-600 px-2 py-1 rounded-md">
                         {item.antibiotic_results.length} آنتی‌بیوتیک
                       </span>
                       <button className="text-slate-400 hover:text-blue-600">
                         <Download size={18} />
                       </button>
                     </div>
                   </div>
                 ))}
               </div>
             )}
          </div>
        ) : (
          <div className="max-w-lg space-y-6">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">نام پزشک / مسئول</label>
                <div className="relative">
                  <User size={18} className="absolute right-3 top-3 text-slate-400" />
                  <input 
                    type="text" 
                    value={profile.name}
                    onChange={(e) => setProfile({...profile, name: e.target.value})}
                    className="w-full pr-10 pl-3 py-2.5 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:outline-none"
                  />
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">نام آزمایشگاه</label>
                <div className="relative">
                  <FlaskConical size={18} className="absolute right-3 top-3 text-slate-400" />
                  <input 
                    type="text" 
                    value={profile.labName}
                    onChange={(e) => setProfile({...profile, labName: e.target.value})}
                    className="w-full pr-10 pl-3 py-2.5 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:outline-none"
                  />
                </div>
              </div>
            </div>

            <div className="pt-6 border-t border-slate-100">
              <h4 className="font-bold text-slate-700 mb-4">وضعیت سیستم</h4>
              <div className="space-y-2">
                 <div className="flex items-center justify-between p-3 bg-green-50 rounded-xl border border-green-100">
                   <div className="flex items-center gap-2">
                     <CheckCircle size={18} className="text-green-600" />
                     <span className="text-sm text-green-800">اتصال به Gemini AI</span>
                   </div>
                   <span className="text-xs font-bold text-green-700">متصل (Key Rotation)</span>
                 </div>
                 <div className="flex items-center justify-between p-3 bg-slate-50 rounded-xl border border-slate-100">
                   <div className="flex items-center gap-2">
                     <Video size={18} className="text-slate-500" />
                     <span className="text-sm text-slate-700">دسترسی دوربین</span>
                   </div>
                   <button className="text-xs text-blue-600 font-medium">تست مجدد</button>
                 </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// 7. Module: Lab Resources (Updated with Batching & QC Voice)
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
          body * {
            visibility: hidden;
          }
          /* Ensure modal content is visible when printing */
          #preview-modal, #preview-modal * {
            visibility: visible;
          }
          #preview-modal {
            position: absolute;
            left: 0;
            top: 0;
            width: 100%;
            height: auto;
            background: white;
            z-index: 9999;
          }
          /* Hide buttons in print mode */
          .no-print {
            display: none !important;
          }
          .disc-preview {
            border: 2px solid black !important;
            color: black !important;
            background: white !important;
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

          {/* Canvas Area (Non-printable, just for working) */}
          <div className="lg:col-span-8 bg-slate-100 rounded-2xl p-8 overflow-y-auto flex items-start justify-center border-2 border-dashed border-slate-200">
             <div className="w-full max-w-[210mm] min-h-[500px] bg-white shadow-sm p-8 flex flex-wrap content-start gap-1 opacity-50 pointer-events-none grayscale">
                <div className="w-full text-center text-slate-400 mb-4 font-bold">نمای کلی (برای چاپ دکمه پیش‌نمایش را بزنید)</div>
                {printQueue.slice(0, 50).map((disc, idx) => (
                  <div key={idx} className="w-8 h-8 border border-slate-300 rounded-full flex items-center justify-center text-[8px]">{disc.code}</div>
                ))}
                {printQueue.length > 50 && <div className="p-2 text-slate-400">... و {printQueue.length - 50} مورد دیگر</div>}
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
                پیش‌نمایش چاپ
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
              <div 
                 id="printable-area" 
                 className="bg-white w-[210mm] min-h-[297mm] p-[10mm] shadow-2xl flex flex-wrap content-start gap-1 mx-auto"
              >
                 {printQueue.map((disc, idx) => (
                    <div 
                      key={idx}
                      className="disc-preview flex items-center justify-center font-black text-black border border-black relative box-border"
                      style={{
                        width: '14mm',  // Tuned for hole punch spacing
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
      )}
    </div>
  );
};

// ... [Main App Component remains mostly same, just updating references] ...

const App = () => {
  const [activeTab, setActiveTab] = useState<Tab>('media-prep');
  const [history, setHistory] = useState<AnalysisResult[]>(() => {
    try {
      const saved = localStorage.getItem('microbio_history');
      return saved ? JSON.parse(saved) : [];
    } catch { return []; }
  });
  
  const [profile, setProfile] = useState<UserProfile>(() => {
    try {
      const saved = localStorage.getItem('microbio_profile');
      return saved ? JSON.parse(saved) : { name: 'دکتر متخصص', labName: 'آزمایشگاه مرکزی', role: 'مسئول فنی' };
    } catch { return { name: 'دکتر متخصص', labName: 'آزمایشگاه مرکزی', role: 'مسئول فنی' }; }
  });

  useEffect(() => {
    localStorage.setItem('microbio_profile', JSON.stringify(profile));
  }, [profile]);

  const addToHistory = (result: AnalysisResult) => {
    const newHistory = [...history, result];
    setHistory(newHistory);
    localStorage.setItem('microbio_history', JSON.stringify(newHistory));
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem('microbio_history');
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'media-prep':
        return <ClinicalConsoleModule />;
      case 'live-lab':
        return <LiveLabModule />;
      case 'analysis':
        return <AnalysisModule onSave={addToHistory} />;
      case 'resources':
        return <LabResourcesModule />;
      case 'settings':
        return <SettingsModule 
          history={history} 
          onClearHistory={clearHistory}
          profile={profile}
          setProfile={setProfile}
        />;
      default:
        return <ClinicalConsoleModule />;
    }
  };

  return (
    <div className="flex h-screen bg-slate-50 text-slate-900 overflow-hidden">
      <aside className="hidden md:flex flex-col w-72 bg-white border-l border-slate-200 h-full shadow-sm z-20">
        <div className="p-6 border-b border-slate-100">
          <div className="flex items-center space-x-3 space-x-reverse text-blue-600">
            <div className="w-10 h-10 bg-blue-600 text-white rounded-xl flex items-center justify-center shadow-lg shadow-blue-200">
              <Microscope size={24} />
            </div>
            <div>
              <h1 className="text-xl font-black tracking-tight">MicroBioMind</h1>
              <span className="text-[10px] bg-blue-50 text-blue-600 px-2 py-0.5 rounded-full font-bold">نسخه ۳.۰ بالینی</span>
            </div>
          </div>
        </div>

        <nav className="flex-1 px-4 py-6 space-y-2 overflow-y-auto">
          <div className="text-xs font-bold text-slate-400 px-4 mb-2">منوی اصلی</div>
          <SidebarItem 
            icon={Stethoscope} 
            label="کنسول بالینی" 
            active={activeTab === 'media-prep'} 
            onClick={() => setActiveTab('media-prep')} 
          />
          <SidebarItem 
            icon={Microscope} 
            label="آزمایشگاه زنده" 
            active={activeTab === 'live-lab'} 
            onClick={() => setActiveTab('live-lab')} 
          />
          <SidebarItem 
            icon={Activity} 
            label="آنالیز و تشخیص" 
            active={activeTab === 'analysis'} 
            onClick={() => setActiveTab('analysis')} 
          />
          
          <div className="my-4 border-t border-slate-100"></div>
          
          <div className="text-xs font-bold text-slate-400 px-4 mb-2">ابزارها</div>
          <SidebarItem 
            icon={Printer} 
            label="منابع و تولیدات" 
            active={activeTab === 'resources'} 
            onClick={() => setActiveTab('resources')} 
          />
          <SidebarItem 
            icon={Settings} 
            label="تنظیمات و پرونده" 
            active={activeTab === 'settings'} 
            onClick={() => setActiveTab('settings')} 
          />
        </nav>

        <div className="p-4 border-t border-slate-100">
          <div className="flex items-center space-x-3 space-x-reverse bg-slate-50 p-3 rounded-xl border border-slate-100">
            <div className="w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center">
              <span className="font-bold text-slate-500 text-xs">Dr</span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-bold text-slate-700 truncate">{profile.name}</p>
              <p className="text-xs text-slate-400 truncate">{profile.role}</p>
            </div>
          </div>
        </div>
      </aside>

      <main className="flex-1 flex flex-col h-full overflow-hidden relative">
        <header className="md:hidden h-16 bg-white border-b border-slate-200 flex items-center justify-between px-4 z-20">
          <div className="flex items-center space-x-2 space-x-reverse text-blue-600">
            <Microscope size={24} />
            <span className="font-bold text-lg">MicroBioMind</span>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-4 md:p-8 scroll-smooth">
          <div className="max-w-6xl mx-auto">
             {renderContent()}
          </div>
        </div>

        <nav className="md:hidden bg-white border-t border-slate-200 pb-safe z-30">
          <div className="flex justify-around items-center px-2">
            <MobileNavItem 
              icon={Stethoscope} 
              label="کنسول" 
              active={activeTab === 'media-prep'} 
              onClick={() => setActiveTab('media-prep')} 
            />
            <MobileNavItem 
              icon={Microscope} 
              label="آزمایشگاه" 
              active={activeTab === 'live-lab'} 
              onClick={() => setActiveTab('live-lab')} 
            />
            <MobileNavItem 
              icon={Printer} 
              label="تولیدات" 
              active={activeTab === 'resources'} 
              onClick={() => setActiveTab('resources')} 
            />
            <MobileNavItem 
              icon={Activity} 
              label="آنالیز" 
              active={activeTab === 'analysis'} 
              onClick={() => setActiveTab('analysis')} 
            />
          </div>
        </nav>
      </main>
    </div>
  );
};

const root = createRoot(document.getElementById('root')!);
root.render(<App />);

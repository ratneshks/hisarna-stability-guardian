import React, { useState, useEffect, useRef, useMemo } from 'react';
import axios from 'axios';
import { 
  LineChart, Line, AreaChart, Area, BarChart, Bar, RadialBarChart, RadialBar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell
} from 'recharts';
import { AlertTriangle, CheckCircle, XCircle, Activity, Info, FileText, Download } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

const COLORS = {
  stable: '#22c55e',
  warning: '#eab308',
  critical: '#ef4444',
  bgDark: '#0f172a',
  cardDark: '#1e293b'
};

export default function App() {
  const [connected, setConnected] = useState(false);
  const [data, setData] = useState(null);
  const [sensorHistory, setSensorHistory] = useState([]);
  const [stabilityHistory, setStabilityHistory] = useState([]);
  const [residualsHistory, setResidualsHistory] = useState([]);
  const [trainingLoss, setTrainingLoss] = useState(null);
  const [timeState, setTimeState] = useState(new Date().toISOString());
  const [demoState, setDemoState] = useState("stable");
  const [demoTimer, setDemoTimer] = useState(0);

  const ws = useRef(null);

  useEffect(() => {
    // Fetch initial history
    axios.get(`${API_URL}/api/training-loss`).then(res => {
      if(res.data && res.data.epoch) {
        const formatted = res.data.epoch.map((e, i) => ({
          epoch: e,
          L_total: res.data.L_total[i],
          L_data: res.data.L_data[i],
          L_NS: res.data.L_NS[i],
          L_HT: res.data.L_HT[i]
        }));
        setTrainingLoss(formatted);
      }
    }).catch(e => console.error(e));

    // WebSocket setup
    const connectWs = () => {
      ws.current = new WebSocket(`${WS_URL}/ws/live-data`);
      ws.current.onopen = () => setConnected(true);
      ws.current.onclose = () => {
        setConnected(false);
        setTimeout(connectWs, 3000);
      };
      ws.current.onmessage = (e) => {
        const payload = JSON.parse(e.data);
        setData(payload);
        setTimeState(payload.timestamp);
        
        // Append to local histories
        setSensorHistory(prev => [...prev.slice(-119), payload.sensors]);
        setStabilityHistory(prev => [...prev.slice(-119), {
          ...payload.stability,
          score: payload.predictions.stability_score
        }]);
        setResidualsHistory(prev => [...prev.slice(-59), payload.physics_residuals]);
      };
    };
    connectWs();
    return () => { if(ws.current) ws.current.close(); };
  }, []);

  const manualReconnect = () => {
    if(ws.current) ws.current.close();
    setConnected(false);
    setData(null);
    // Connection will be re-established by the onclose logic or by a fresh connectWs call
    // but for immediate action we can just reload
    window.location.reload();
  };

  useEffect(() => {
    const int = setInterval(() => setDemoTimer(p => p + 1), 1000);
    return () => clearInterval(int);
  }, [demoState]);

  const triggerScenario = async (mode) => {
    try {
      await axios.post(`${API_URL}/api/trigger-scenario`, { scenario: mode });
      setDemoState(mode);
      setDemoTimer(0);
    } catch (e) { console.error(e); }
  };

  const exportReport = () => {
    const text = `HIsarna Stability Guardian - Export
Time: ${timeState}
Status: ${data?.stability?.status}
Mahalanobis Distance: ${data?.stability?.mahalanobis_distance?.toFixed(2)}
Recommendation: ${data?.recommendation}

Last Sensor Readings:
${Object.entries(data?.sensors || {}).map(([k,v]) => `${k}: ${v.toFixed(3)}`).join('\n')}
`;
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `hisarna_report_${Date.now()}.txt`;
    a.click();
  };

  if (!data) return (
    <div className="flex flex-col h-screen items-center justify-center bg-slate-900 text-white font-mono space-y-6">
      <div className="flex items-center space-x-3">
        <div className="h-4 w-4 bg-blue-500 animate-ping rounded-full"></div>
        <div className="text-xl">Connecting to HIsarna Core...</div>
      </div>
      <div className="text-slate-500 text-xs text-center max-w-md">
        If this takes more than 30 seconds, the backend might be waking up from cold sleep.
      </div>
      <button 
        onClick={() => window.location.reload()}
        className="px-6 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-600 rounded text-sm transition-all"
      >
        Retry Connection
      </button>
    </div>
  );

  const { sensors, predictions, stability, physics_residuals, recommendation, active_alerts } = data;

  const pulseClass = stability.status === 'STABLE' ? 'pulse-green' : (stability.status === 'WARNING' ? 'pulse-yellow' : 'pulse-red');
  const statusColor = stability.status === 'STABLE' ? COLORS.stable : (stability.status === 'WARNING' ? COLORS.warning : COLORS.critical);

  // Derive Sensitivity (Mock interpretability based on predefined intuition for demo)
  const sensitivityData = [
    { name: 'phi_O2', value: stability.status !== 'STABLE' ? 0.8 : 0.2 },
    { name: 'T_cyclone', value: stability.status !== 'STABLE' ? 0.7 : 0.3 },
    { name: 'm_ore', value: stability.status === 'WARNING' ? 0.6 : 0.1 },
    { name: 'vibration', value: stability.status === 'CRITICAL' ? 0.9 : 0.1 }
  ];

  return (
    <div className="flex h-screen bg-slate-900 text-slate-200 overflow-hidden text-sm">
      {/* Sidebar */}
      <div className="w-64 bg-slate-800 border-r border-slate-700 flex flex-col z-20 shadow-xl">
        <div className="p-4 border-b border-slate-700 bg-slate-900 flex items-center justify-between">
          <div className="font-bold text-red-600 tracking-wider">TATA STEEL</div>
          <div className={`h-3 w-3 rounded-full ${connected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
        </div>
        
        <div className="p-4 border-b border-slate-700">
          <h2 className="text-xs text-slate-400 uppercase font-bold mb-2">Demo Controls</h2>
          <div className="space-y-2">
            <button onClick={() => triggerScenario('stable')} className={`w-full py-2 rounded font-bold ${demoState==='stable' ? 'bg-green-600 text-white' : 'bg-slate-700 hover:bg-slate-600'}`}>🟢 STABLE</button>
            <button onClick={() => triggerScenario('warning')} className={`w-full py-2 rounded font-bold ${demoState==='warning' ? 'bg-yellow-600 text-white' : 'bg-slate-700 hover:bg-slate-600'}`}>🟡 WARNING</button>
            <button onClick={() => triggerScenario('critical')} className={`w-full py-2 rounded font-bold ${demoState==='critical' ? 'bg-red-600 text-white' : 'bg-slate-700 hover:bg-slate-600'}`}>🔴 CRITICAL</button>
          </div>
          <div className="text-xs text-center mt-2 text-slate-400">Scenario active for: {demoTimer}s</div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <div>
            <h2 className="text-xs text-slate-400 uppercase font-bold mb-2 flex items-center justify-between">
              Active Alerts <span className="bg-slate-700 px-2 rounded-full">{active_alerts.length}</span>
            </h2>
            <div className="space-y-2">
              {active_alerts.length === 0 && <div className="text-xs text-slate-500">No active alerts.</div>}
              {active_alerts.map(a => (
                <div key={a.id} className={`p-2 rounded text-xs border-l-2 ${a.severity === 'CRITICAL' ? 'border-red-500 bg-red-900/20 text-red-200' : 'border-yellow-500 bg-yellow-900/20 text-yellow-200'}`}>
                  <div className="font-bold">{a.severity}</div>
                  <div>{a.message}</div>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h2 className="text-xs text-slate-400 uppercase font-bold mb-2">Sensor Quick-View</h2>
            <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-xs">
              {Object.entries(sensors).map(([k,v]) => (
                <React.Fragment key={k}>
                  <div className="text-slate-500 truncate" title={k}>{k.replace('phi_','').replace('_rms','')}</div>
                  <div className="text-right font-mono text-slate-300">{v.toFixed(2)}</div>
                </React.Fragment>
              ))}
            </div>
          </div>
          
          <div>
            <h2 className="text-xs text-slate-400 uppercase font-bold mb-2">Governing Equations</h2>
            <div className="bg-slate-900 p-2 rounded text-[10px] font-mono text-blue-300 opacity-80">
              <div>∇·u = 0</div>
              <div className="mt-1">ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u</div>
              <div className="mt-1">ρCp(∂T/∂t + u·∇T) = k∇²T + Q</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-full overflow-hidden bg-slate-900">
        <header className="h-14 bg-slate-800 border-b border-slate-700 flex items-center justify-between px-6 shadow-sm z-10">
          <div className="text-lg font-bold">HIsarna Stability Guardian (PINN)</div>
          <div className="flex items-center space-x-4">
            <button onClick={exportReport} className="flex items-center space-x-1 text-xs bg-slate-700 hover:bg-slate-600 px-3 py-1.5 rounded transition">
              <Download size={14} /> <span>Export Report</span>
            </button>
            <div className="text-xs text-slate-400 font-mono">{new Date(timeState).toLocaleTimeString()}</div>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          
          {/* Row 1: KPIs */}
          <div className="grid grid-cols-4 gap-4">
            {/* Status Card */}
            <div className={`bg-slate-800 p-4 rounded-lg border border-slate-700 flex flex-col justify-center items-center ${pulseClass} transition-all duration-500`}>
              <div className="text-xs text-slate-400 uppercase font-bold mb-2">System Status</div>
              <div className="text-3xl font-bold mb-2" style={{color: statusColor}}>{stability.status}</div>
              <div className="w-full bg-slate-900 rounded-full h-2.5 mb-1 relative overflow-hidden">
                <div className="h-2.5 rounded-full transition-all duration-300" style={{width: `${Math.min(100, stability.normalized_distance * 50)}%`, backgroundColor: statusColor}}></div>
                <div className="absolute top-0 bottom-0 left-[30%] w-0.5 bg-slate-500/50"></div> {/* 0.6 mark */}
                <div className="absolute top-0 bottom-0 left-[50%] w-0.5 bg-red-500/50"></div> {/* 1.0 mark */}
              </div>
              <div className="text-[10px] text-slate-500">Mahalanobis: {stability.normalized_distance.toFixed(2)}</div>
            </div>

            {/* Temp Card */}
            <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 flex flex-col">
              <div className="text-xs text-slate-400 uppercase font-bold mb-1">Cyclone Temp</div>
              <div className="text-3xl font-mono font-bold mt-auto mb-1 text-slate-100">{sensors.T_cyclone.toFixed(1)}<span className="text-lg text-slate-500">°C</span></div>
              <div className="w-full bg-slate-900 rounded h-1.5 overflow-hidden">
                <div className={`h-full ${sensors.T_cyclone > 1500 ? 'bg-red-500' : (sensors.T_cyclone > 1480 ? 'bg-yellow-500' : 'bg-blue-400')}`} style={{width: `${Math.min(100, (sensors.T_cyclone - 1300)/3)}%`}}></div>
              </div>
            </div>

            {/* O2 Card */}
            <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 flex flex-col relative">
               <div className="text-xs text-slate-400 uppercase font-bold mb-1 absolute top-4 left-4 z-10">Oxygen Fraction</div>
               <div className="absolute inset-0 flex items-center justify-center mt-4">
                 <div className="text-2xl font-mono font-bold" style={{color: sensors.phi_O2 < 1 ? '#ef4444' : '#22c55e'}}>{sensors.phi_O2.toFixed(1)}%</div>
               </div>
               <ResponsiveContainer width="100%" height="100%">
                 <RadialBarChart cx="50%" cy="50%" innerRadius="70%" outerRadius="90%" barSize={10} data={[{name: 'O2', value: sensors.phi_O2, fill: sensors.phi_O2 < 1 ? '#ef4444' : '#22c55e'}]} startAngle={180} endAngle={0}>
                   <RadialBar background dataKey="value" cornerRadius={5} />
                 </RadialBarChart>
               </ResponsiveContainer>
            </div>

            {/* Physics Compliance */}
            <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 flex flex-col">
              <div className="text-xs text-slate-400 uppercase font-bold mb-2">Physics Compliance</div>
              <div className="flex-1 flex items-end justify-around pb-2">
                 <div className="w-1/3 flex flex-col items-center">
                    <div className="w-full bg-slate-900 rounded-t h-16 relative flex items-end justify-center">
                       <div className="w-3/4 rounded-t transition-all bg-indigo-500" style={{height: `${Math.max(5, 100 - physics_residuals.L_NS * 1000)}%`}}></div>
                    </div>
                    <div className="text-[10px] mt-1 text-slate-400">N-S</div>
                 </div>
                 <div className="w-1/3 flex flex-col items-center">
                    <div className="w-full bg-slate-900 rounded-t h-16 relative flex items-end justify-center">
                       <div className="w-3/4 rounded-t transition-all bg-purple-500" style={{height: `${Math.max(5, 100 - Math.min(100, physics_residuals.L_HT * 1e9))}%`}}></div>
                    </div>
                    <div className="text-[10px] mt-1 text-slate-400">Heat</div>
                 </div>
              </div>
            </div>
          </div>

          {/* Row 2: Charts */}
          <div className="grid grid-cols-2 gap-4 h-64">
             <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 flex flex-col">
               <div className="text-xs text-slate-400 uppercase font-bold mb-2">Sensor Trends (Normalized View)</div>
               <div className="flex-1 min-h-0">
                 <ResponsiveContainer width="100%" height="100%">
                   <LineChart data={sensorHistory} margin={{top:5, right:5, bottom:5, left:-20}}>
                     <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                     <XAxis dataKey={(v,i) => i} tick={false} stroke="#64748b" />
                     <YAxis stroke="#64748b" domain={['dataMin - 10', 'dataMax + 10']} tickFormatter={v => ''} />
                     <Tooltip contentStyle={{backgroundColor:'#1e293b', borderColor:'#334155', color:'#f8fafc'}} />
                     <Legend iconType="plainline" wrapperStyle={{fontSize:'10px'}} />
                     <Line type="monotone" dataKey="T_cyclone" stroke="#ef4444" dot={false} isAnimationActive={false} />
                     <Line type="monotone" dataKey="phi_O2" stroke="#3b82f6" dot={false} isAnimationActive={false} />
                     <Line type="monotone" dataKey="m_ore" stroke="#8b5cf6" dot={false} isAnimationActive={false} />
                   </LineChart>
                 </ResponsiveContainer>
               </div>
             </div>

             <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 flex flex-col">
               <div className="text-xs text-slate-400 uppercase font-bold mb-2">Stability Trajectory (Mahalanobis Distance)</div>
               <div className="flex-1 min-h-0">
                 <ResponsiveContainer width="100%" height="100%">
                   <AreaChart data={stabilityHistory} margin={{top:5, right:5, bottom:5, left:-20}}>
                     <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                     <XAxis dataKey={(v,i) => i} tick={false} stroke="#64748b" />
                     <YAxis stroke="#64748b" domain={[0, 1.5]} ticks={[0, 0.6, 1.0, 1.5]} />
                     <Tooltip contentStyle={{backgroundColor:'#1e293b', borderColor:'#334155', color:'#f8fafc'}} />
                     {/* Safe zone */}
                     <Area type="monotone" dataKey="normalized_distance" stroke={statusColor} fill={statusColor} fillOpacity={0.3} isAnimationActive={false} />
                   </AreaChart>
                 </ResponsiveContainer>
               </div>
             </div>
          </div>

          {/* Row 3: Bottom Panels */}
          <div className="grid grid-cols-3 gap-4 h-56">
            
            <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 flex flex-col">
               <div className="text-xs text-slate-400 uppercase font-bold mb-1">Real-time Physics Compliance</div>
               <div className="text-[10px] text-slate-500 mb-2">Lower values = strict adherence to laws of physics</div>
               <div className="flex-1 min-h-0">
                 <ResponsiveContainer width="100%" height="100%">
                   <LineChart data={residualsHistory} margin={{top:5, right:5, bottom:5, left:0}}>
                     <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                     <XAxis dataKey={(v,i) => i} tick={false} />
                     <YAxis scale="log" domain={['auto', 'auto']} stroke="#64748b" tick={{fontSize: 10}} />
                     <Tooltip contentStyle={{backgroundColor:'#1e293b', borderColor:'#334155'}} />
                     <Legend wrapperStyle={{fontSize:'10px'}}/>
                     <Line type="stepAfter" dataKey="L_NS" name="Navier-Stokes" stroke="#6366f1" dot={false} isAnimationActive={false} />
                   </LineChart>
                 </ResponsiveContainer>
               </div>
            </div>

            <div className={`bg-slate-800 p-4 rounded-lg border-2 flex flex-col relative ${stability.status === 'STABLE' ? 'border-green-500/50' : (stability.status === 'WARNING' ? 'border-yellow-500/50' : 'border-red-500/50')}`}>
               <div className="text-xs text-slate-400 uppercase font-bold mb-4">Operator Recommendation</div>
               <div className="flex items-start space-x-3">
                 {stability.status === 'STABLE' && <CheckCircle className="text-green-500 flex-shrink-0" size={24} />}
                 {stability.status === 'WARNING' && <AlertTriangle className="text-yellow-500 flex-shrink-0" size={24} />}
                 {stability.status === 'CRITICAL' && <XCircle className="text-red-500 flex-shrink-0" size={24} />}
                 <div className="text-sm font-medium leading-relaxed">{recommendation}</div>
               </div>
               <div className="mt-auto text-xs text-slate-500">
                 PINN Action Engine • Confidence: {(predictions.stability_score * 100).toFixed(1)}%
               </div>
            </div>

            <div className="bg-slate-800 p-4 rounded-lg border border-slate-700 flex flex-col">
               <div className="text-xs text-slate-400 uppercase font-bold mb-1">Model Training History</div>
               <div className="flex-1 min-h-0">
                 {trainingLoss ? (
                   <ResponsiveContainer width="100%" height="100%">
                     <LineChart data={trainingLoss} margin={{top:5, right:5, bottom:5, left:0}}>
                       <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                       <XAxis dataKey="epoch" tick={{fontSize:10}} stroke="#64748b" />
                       <YAxis scale="log" domain={['auto', 'auto']} tick={{fontSize:10}} stroke="#64748b" />
                       <Tooltip contentStyle={{backgroundColor:'#1e293b', borderColor:'#334155'}} />
                       <Line type="monotone" dataKey="L_total" name="Total Loss" stroke="#e2e8f0" dot={false} strokeWidth={2} />
                       <Line type="monotone" dataKey="L_BC" name="Boundary Loss" stroke="#f59e0b" dot={false} />
                     </LineChart>
                   </ResponsiveContainer>
                 ) : (
                   <div className="flex items-center justify-center h-full text-xs text-slate-500">Loading training curves...</div>
                 )}
               </div>
            </div>

          </div>

        </div>
      </div>
    </div>
  );
}

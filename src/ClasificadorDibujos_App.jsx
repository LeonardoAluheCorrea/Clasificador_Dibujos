// ClasificadorDibujos_App.jsx
// Aplicación React + TensorFlow.js para el Trabajo PrÁctico: "Clasificador de Dibujos"
// - Interfaz para crear categorías, capturar imágenes desde la cámara, almacenar dataset en localStorage
// - Entrena una red neuronal pequeña (convolucional simple) usando @tensorflow/tfjs en el navegador
// - Muestra progreso de entrenamiento y predicciones en tiempo real
// - Visualización simple de la red (nodos que "se iluminan")
// Requisitos: proyecto React (Vite o Create React App). Instalar dependencias:
// npm install @tensorflow/tfjs react-webcam
// (Opcional: tailwindcss si quieres estilizar con Tailwind)

import React, { useRef, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs';

// Parámetros configurables
const IMAGE_SIZE = 64; // dimensiones a las que redimensionamos las imágenes
const CHANNELS = 3;
const BATCH_SIZE = 16;
const EPOCHS = 15;

export default function ClasificadorDibujosApp() {
  const webcamRef = useRef(null);
  const [categories, setCategories] = useState(['Gato','Casa','Sol']);
  const [currentCat, setCurrentCat] = useState('Gato');
  const [dataset, setDataset] = useState({}); // {cat: [base64 images]}
  const [model, setModel] = useState(null);
  const [training, setTraining] = useState(false);
  const [logs, setLogs] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [visualStep, setVisualStep] = useState(null);

  useEffect(()=>{
    // cargar dataset desde localStorage si existe
    const saved = localStorage.getItem('clasificador_dataset_v1');
    if(saved) setDataset(JSON.parse(saved));
  },[]);

  useEffect(()=>{
    // persist dataset
    localStorage.setItem('clasificador_dataset_v1', JSON.stringify(dataset));
  },[dataset]);

  // captura imagen desde la webcam y la guarda como base64 en la categoría actual
  const capture = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    if(!imageSrc) return;
    setDataset(prev => {
      const copy = {...prev};
      if(!copy[currentCat]) copy[currentCat]=[];
      copy[currentCat].push(imageSrc);
      return copy;
    });
  }

  const addCategory = () => {
    const name = prompt('Nombre de la nueva categoría:');
    if(!name) return;
    setCategories(c=> (c.includes(name)? c : [...c,name]));
    setCurrentCat(name);
  }

  const clearDataset = () => {
    if(!confirm('Borrar todo el dataset?')) return;
    setDataset({});
  }

  // Preprocesado: convierte array de base64 images en tensores X e Y
  const buildTensorsFromDataset = async () => {
    const cats = Object.keys(dataset).filter(k=>dataset[k].length>0);
    if(cats.length<2) throw new Error('Se necesitan al menos 2 categorías con imágenes.');

    const xs = [];
    const ys = [];
    for(let i=0;i<cats.length;i++){
      const cat = cats[i];
      for(const b64 of dataset[cat]){
        const img = await loadImage(b64);
        const tensor = tf.browser.fromPixels(img).resizeNearestNeighbor([IMAGE_SIZE,IMAGE_SIZE]).toFloat().div(255.0);
        xs.push(tensor);
        ys.push(i);
      }
    }
    const X = tf.stack(xs);
    const Y = tf.tensor1d(ys,'int32');
    const Y_onehot = tf.oneHot(Y, cats.length);
    return {X,Y:Y_onehot, labels:cats};
  }

  const loadImage = (dataUrl) => {
    return new Promise((resolve)=>{
      const img = new Image();
      img.src = dataUrl;
      img.onload = ()=>resolve(img);
    });
  }

  const createModel = (numClasses) => {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({inputShape:[IMAGE_SIZE,IMAGE_SIZE,CHANNELS], filters:16, kernelSize:3, activation:'relu'}));
    model.add(tf.layers.maxPooling2d({poolSize:2}));
    model.add(tf.layers.conv2d({filters:32, kernelSize:3, activation:'relu'}));
    model.add(tf.layers.maxPooling2d({poolSize:2}));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({units:64, activation:'relu'}));
    model.add(tf.layers.dense({units:numClasses, activation:'softmax'}));

    model.compile({optimizer: tf.train.adam(0.001), loss:'categoricalCrossentropy', metrics:['accuracy']});

    return model;
  }

  const trainModel = async () => {
    setLogs([]);
    setTraining(true);
    setVisualStep('compiling');
    try{
      const {X,Y,labels} = await buildTensorsFromDataset();
      setVisualStep('created_tensors');
      await tf.nextFrame();
      const m = createModel(labels.length);
      setModel(m);
      setVisualStep('training');

      await m.fit(X,Y,{
        epochs: EPOCHS,
        batchSize: BATCH_SIZE,
        shuffle:true,
        callbacks: {
          onEpochEnd: async (epoch, logsEpoch) => {
            setLogs(prev=>[...prev, {epoch:epoch+1, ...logsEpoch}]);
            setVisualStep({phase:'epoch', epoch:epoch+1});
            await tf.nextFrame();
          }
        }
      });

      setVisualStep('trained');
      X.dispose(); Y.dispose();
      alert('Entrenamiento terminado');
    } catch(err){
      alert('Error durante el entrenamiento: '+err.message);
    } finally{
      setTraining(false);
      setVisualStep(null);
    }
  }

  // Predicción en tiempo real desde la webcam
  const predictOnce = async () => {
    if(!model){ alert('Entrena el modelo primero'); return; }
    const imageSrc = webcamRef.current.getScreenshot();
    const img = await loadImage(imageSrc);
    const tensor = tf.browser.fromPixels(img).resizeNearestNeighbor([IMAGE_SIZE,IMAGE_SIZE]).toFloat().div(255.0).expandDims(0);
    const preds = model.predict(tensor);
    const arr = await preds.data();
    tensor.dispose(); preds.dispose();
    // obtener labels del dataset
    const labels = Object.keys(dataset).filter(k=>dataset[k].length>0);
    const pairs = labels.map((l,i)=>({label:l,prob:arr[i]}));
    pairs.sort((a,b)=>b.prob-a.prob);
    setPrediction(pairs);
    setVisualStep({phase:'predict'});
  }

  const exportDataset = () => {
    const blob = new Blob([JSON.stringify(dataset)], {type:'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'dataset_clasificador.json'; a.click();
    URL.revokeObjectURL(url);
  }

  const importDataset = (ev) => {
    const f = ev.target.files[0];
    if(!f) return;
    const r = new FileReader();
    r.onload = (e)=>{
      try{
        const obj = JSON.parse(e.target.result);
        setDataset(obj);
        alert('Dataset importado');
      }catch(err){ alert('Error leyendo archivo'); }
    }
    r.readAsText(f);
  }

  // Red visual simple: Circulos para capaz y nodos que se iluminan al entrenar
  const NetworkViz = ({step}) => {
  if (!step) step = null; // prevenir errores si no hay estado aún
  const active = typeof step === 'object' && step !== null ? step.phase : step;

    return (
      <div className="p-2 border rounded">
        <div className="text-sm mb-2">Visualización simple de la red</div>
        <svg width="300" height="120">
          {/* input layer */}
          {[0,1,2,3].map(i=> (
            <circle key={i} cx={40} cy={20+i*22} r={8} fill={active==='training' || active==='created_tensors' ? '#ffd' : '#eee'} stroke="#333" />
          ))}
          {/* hidden layer */}
          {[0,1,2].map(i=> (
            <circle key={'h'+i} cx={150} cy={20+i*28} r={10} fill={active && active.phase==='epoch' ? '#ffa' : '#eee'} stroke="#333" />
          ))}
          {/* output layer */}
          <circle cx={260} cy={45} r={12} fill={active && (active==='trained' || (active.phase==='predict')) ? '#ffb' : '#eee'} stroke="#333" />
        </svg>
      </div>
    )
  }

  return (
    <div className="p-4 max-w-4xl mx-auto font-sans">
      <h1 className="text-2xl font-bold mb-3">Clasificador de Dibujos (React + TensorFlow.js)</h1>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="mb-2">Cámara</div>
          <Webcam audio={false} screenshotFormat="image/jpeg" ref={webcamRef} videoConstraints={{facingMode:'user'}} width={320} height={240} />
          <div className="mt-2 flex gap-2">
            <select value={currentCat} onChange={e=>setCurrentCat(e.target.value)}>
              {categories.map(c=> <option key={c} value={c}>{c}</option>)}
            </select>
            <button onClick={capture} className="px-2 py-1 border rounded">Capturar</button>
            <button onClick={addCategory} className="px-2 py-1 border rounded">Nueva categoría</button>
          </div>

          <div className="mt-3">
            <strong>Dataset (por categoría)</strong>
            <div className="max-h-40 overflow-auto border p-2 mt-1">
              {Object.keys(dataset).length===0 && <div className="text-sm text-gray-500">(vacío)</div>}
              {Object.entries(dataset).map(([k,arr])=> (
                <div key={k} className="mb-2">
                  <div className="font-semibold">{k} — {arr.length} imágenes</div>
                  <div className="flex gap-1 overflow-auto">{arr.slice(-6).map((b64,i)=> <img key={i} src={b64} style={{width:50,height:50,objectFit:'cover'}} alt="m"/>)}</div>
                </div>
              ))}
            </div>
            <div className="mt-2 flex gap-2">
              <button onClick={trainModel} disabled={training} className="px-3 py-1 border rounded">Entrenar</button>
              <button onClick={predictOnce} className="px-3 py-1 border rounded">Probar (predecir)</button>
              <button onClick={exportDataset} className="px-3 py-1 border rounded">Exportar dataset</button>
              <label className="px-3 py-1 border rounded cursor-pointer">
                Importar
                <input type="file" accept="application/json" onChange={importDataset} style={{display:'none'}}/>
              </label>
              <button onClick={clearDataset} className="px-3 py-1 border rounded">Borrar dataset</button>
            </div>
          </div>
        </div>

        <div>
          <div className="mb-2">Entrenamiento y métricas</div>
          <div className="border p-2 mb-2 h-48 overflow-auto">
            {logs.length===0 && <div className="text-sm text-gray-500">Aquí aparecerán los logs de las épocas de entrenamiento.</div>}
            {logs.map((l,i)=> (
              <div key={i}>Época {l.epoch}: loss={l.loss.toFixed(3)} acc={ (l.acc||l.accuracy||0).toFixed(3)}</div>
            ))}
          </div>

          <NetworkViz step={visualStep} />

          <div className="mt-3">
            <strong>Predicción</strong>
            <div className="border p-2 mt-1">
              {!prediction && <div className="text-sm text-gray-500">Realiza una predicción para ver los porcentajes.</div>}
              {prediction && (
                <div>
                  {prediction.map((p,i)=> (
                    <div key={i}>{p.label}: {(p.prob*100).toFixed(1)}%</div>
                  ))}
                </div>
              )}
            </div>
          </div>

        </div>
      </div>

      <div className="mt-4 text-sm text-gray-600">
        <strong>Notas:</strong>
        <ul className="list-disc ml-5">
          <li>Este ejemplo entrena la red en el navegador usando TensorFlow.js. Para datasets grandes o entrenamientos más serios, entrena en el servidor o con Colab y carga el modelo al cliente.</li>
          <li>La visualización de la red es intencionalmente simple: es didáctica para primaria. Puedes mejorarla conectando los nodos a métricas o activaciones reales.</li>
          <li>Si quieres que genere un repositorio completo (estructura, package.json, README, scripts de despliegue), dímelo y lo creo en la siguiente respuesta.</li>
        </ul>
      </div>
    </div>
  )
}

// ClasificadorDibujos_App.jsx
// React + TensorFlow.js – versión con visualización de activaciones reales
// Leonardo A. Correa, 2025

import React, { useRef, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs';

const IMAGE_SIZE = 64;
const CHANNELS = 3;
const BATCH_SIZE = 16;
const EPOCHS = 15;

export default function ClasificadorDibujosApp() {
  const webcamRef = useRef(null);
  const [categories, setCategories] = useState(['Gato','Casa','Sol']);
  const [currentCat, setCurrentCat] = useState('Gato');
  const [dataset, setDataset] = useState({});
  const [model, setModel] = useState(null);
  const [activationModel, setActivationModel] = useState(null);
  const [training, setTraining] = useState(false);
  const [logs, setLogs] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [activations, setActivations] = useState([]);

  useEffect(()=>{
    const saved = localStorage.getItem('clasificador_dataset_v1');
    if(saved) setDataset(JSON.parse(saved));
  },[]);

  useEffect(()=>{
    localStorage.setItem('clasificador_dataset_v1', JSON.stringify(dataset));
  },[dataset]);

  const capture = () => {
    if (!webcamRef.current) return alert("Cámara no lista");
    const imageSrc = webcamRef.current.getScreenshot();
    if(!imageSrc) return alert("No se pudo capturar la imagen");
    setDataset(prev => {
      const copy = {...prev};
      if(!copy[currentCat]) copy[currentCat]=[];
      copy[currentCat].push(imageSrc);
      return copy;
    });
  };

  const addCategory = () => {
    const name = prompt('Nombre de la nueva categoría:');
    if(!name) return;
    setCategories(c=> (c.includes(name)? c : [...c,name]));
    setCurrentCat(name);
  };

  const clearDataset = () => {
    if(!confirm('Borrar todo el dataset?')) return;
    setDataset({});
  };

  const loadImage = (dataUrl) => {
    return new Promise((resolve)=>{
      const img = new Image();
      img.src = dataUrl;
      img.onload = ()=>resolve(img);
    });
  };

  const buildTensorsFromDataset = async () => {
    const cats = Object.keys(dataset).filter(k=>dataset[k].length>0);
    if(cats.length<2) throw new Error('Se necesitan al menos 2 categorías con imágenes.');

    const xs = [];
    const ys = [];
    for(let i=0;i<cats.length;i++){
      const cat = cats[i];
      for(const b64 of dataset[cat]){
        const img = await loadImage(b64);
        const tensor = tf.browser.fromPixels(img)
          .resizeNearestNeighbor([IMAGE_SIZE,IMAGE_SIZE])
          .toFloat()
          .div(255.0);
        xs.push(tensor);
        ys.push(i);
      }
    }
    const X = tf.stack(xs);
    const Y = tf.tensor1d(ys,'int32');
    const Y_onehot = tf.oneHot(Y, cats.length);
    return {X,Y:Y_onehot, labels:cats};
  };

  const createModel = (numClasses) => {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({inputShape:[IMAGE_SIZE,IMAGE_SIZE,CHANNELS], filters:16, kernelSize:3, activation:'relu', name:'conv1'}));
    model.add(tf.layers.maxPooling2d({poolSize:2}));
    model.add(tf.layers.conv2d({filters:32, kernelSize:3, activation:'relu', name:'conv2'}));
    model.add(tf.layers.maxPooling2d({poolSize:2}));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({units:64, activation:'relu', name:'dense'}));
    model.add(tf.layers.dense({units:numClasses, activation:'softmax', name:'output'}));
    model.compile({optimizer: tf.train.adam(0.001), loss:'categoricalCrossentropy', metrics:['accuracy']});
    return model;
  };

  const trainModel = async () => {
    setLogs([]);
    setTraining(true);
    try{
      const {X,Y,labels} = await buildTensorsFromDataset();
      const m = createModel(labels.length);
      setModel(m);

      // modelo auxiliar para capturar activaciones
      const activationM = tf.model({inputs: m.input, outputs: m.layers.map(l=>l.output)});
      setActivationModel(activationM);

      await m.fit(X,Y,{
        epochs: EPOCHS,
        batchSize: BATCH_SIZE,
        shuffle:true,
        callbacks: {
          onEpochEnd: async (epoch, logsEpoch) => {
            // Tomamos una imagen del dataset para visualizar activaciones
            const sample = X.slice([0,0,0,0],[1,IMAGE_SIZE,IMAGE_SIZE,CHANNELS]);
            const acts = activationM.predict(sample);
            const arr = await Promise.all(acts.map(async a=>{
              const mean = (await a.mean().data())[0];
              a.dispose();
              return mean;
            }));
            setActivations(arr);
            setLogs(prev=>[...prev, {epoch:epoch+1, ...logsEpoch}]);
            await tf.nextFrame();
          }
        }
      });

      X.dispose(); Y.dispose();
      alert('Entrenamiento terminado');
    } catch(err){
      alert('Error durante el entrenamiento: '+err.message);
    } finally{
      setTraining(false);
    }
  };

  const predictOnce = async () => {
    if(!model || !activationModel){ alert('Entrena el modelo primero'); return; }
    const imageSrc = webcamRef.current.getScreenshot();
    const img = await loadImage(imageSrc);
    const tensor = tf.browser.fromPixels(img)
      .resizeNearestNeighbor([IMAGE_SIZE,IMAGE_SIZE])
      .toFloat()
      .div(255.0)
      .expandDims(0);

    const acts = activationModel.predict(tensor);
    const arrActs = await Promise.all(acts.map(async a=>{
      const mean = (await a.mean().data())[0];
      a.dispose();
      return mean;
    }));
    console.log("Activaciones promedio por capa:", arrActs);

    setActivations(arrActs);

    const preds = model.predict(tensor);
    const arr = await preds.data();
    const labels = Object.keys(dataset).filter(k=>dataset[k].length>0);
    const pairs = labels.map((l,i)=>({label:l,prob:arr[i]}));
    pairs.sort((a,b)=>b.prob-a.prob);
    setPrediction(pairs);

    tensor.dispose(); preds.dispose();
  };

  const exportDataset = () => {
    const blob = new Blob([JSON.stringify(dataset)], {type:'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'dataset_clasificador.json'; a.click();
    URL.revokeObjectURL(url);
  };

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
  };

// --- NetworkViz: disposición horizontal forzada de izquierda a derecha ---
const NetworkViz = ({ activations = [], prediction = [] }) => {
  const layerNames = ['Conv1', 'Pool1', 'Conv2', 'Pool2', 'Densa', 'Softmax'];
  const normalize = (x) => Math.min(1, Math.max(0, x / 0.8));
  const colors = (v) => {
    const r = Math.round(255 * v);
    const g = Math.round(100 + 100 * (1 - v));
    return `rgba(${r}, ${g}, 0, ${0.8})`;
  };

  return (
    <div
      className="w-full border rounded mt-3"
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        overflowX: 'auto',
      }}
    >
      <h3 className="text-sm font-semibold my-2 text-center">
        Visualización de la red neuronal (horizontal)
      </h3>

      {/* fila horizontal real */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'flex-start',
          justifyContent: 'flex-start',
          gap: '60px',
          minWidth: '1000px', // asegura disposición horizontal
          padding: '10px',
        }}
      >
        {layerNames.map((name, i) => {
          const a = normalize(activations[i] || 0);

          // capas convolucionales
          if (name.startsWith('Conv')) {
            const filters = 12;
            return (
              <div key={name} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <div className="text-xs mb-1 font-semibold">{name}</div>
                <div
                  style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    width: '70px',
                    gap: '2px',
                    transform: `scale(${0.8 + 0.3 * a})`,
                    transition: 'transform 1.2s ease-in-out',
                  }}
                >
                  {Array.from({ length: filters }).map((_, j) => (
                    <div
                      key={j}
                      style={{
                        width: '18px',
                        height: '18px',
                        borderRadius: '3px',
                        backgroundColor: colors(a * (0.6 + 0.4 * Math.random())),
                        transition: 'background-color 1.2s ease-in-out',
                      }}
                    />
                  ))}
                </div>
              </div>
            );
          }

          // pooling
          if (name.startsWith('Pool')) {
            return (
              <div key={name} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <div className="text-xs mb-1 font-semibold">{name}</div>
                <div
                  style={{
                    width: '18px',
                    height: `${40 + 100 * a}px`,
                    backgroundColor: colors(a),
                    borderRadius: '6px',
                    transition: 'height 1.2s ease-in-out, background-color 1.2s ease-in-out',
                  }}
                />
              </div>
            );
          }

          // densa
          if (name === 'Densa') {
            return (
              <div key={name} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <div className="text-xs mb-1 font-semibold">{name}</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {Array.from({ length: 6 }).map((_, j) => (
                    <div
                      key={j}
                      style={{
                        width: '22px',
                        height: '22px',
                        borderRadius: '50%',
                        backgroundColor: colors(a * (0.7 + 0.3 * Math.random())),
                        transition: 'background-color 1.2s ease-in-out',
                      }}
                    />
                  ))}
                </div>
              </div>
            );
          }

          // salida
          if (name === 'Softmax' && prediction?.length > 0) {
            return (
              <div key={name} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <div className="text-xs mb-1 font-semibold">Salida</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {prediction.map((p, j) => (
                    <div
                      key={j}
                      title={`${p.label}: ${(p.prob * 100).toFixed(1)}%`}
                      style={{
                        width: '26px',
                        height: '26px',
                        borderRadius: '50%',
                        border: '1px solid #333',
                        backgroundColor: colors(p.prob),
                        transition: 'background-color 1.2s ease-in-out',
                      }}
                    />
                  ))}
                </div>
              </div>
            );
          }

          return null;
        })}
      </div>
    </div>
  );
};


  return (
    <div className="p-4 max-w-4xl mx-auto font-sans">
      <h1 className="text-2xl font-bold mb-3">Clasificador de Dibujos (React + TensorFlow.js)</h1>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="mb-2">Cámara</div>
          <Webcam audio={false} screenshotFormat="image/jpeg"
            ref={webcamRef} videoConstraints={{facingMode:'user'}}
            width={320} height={240}
          />
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

          <NetworkViz activations={activations} />

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
    </div>
  );
}

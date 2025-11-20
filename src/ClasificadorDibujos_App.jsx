// ClasificadorDibujos_App.jsx
// React + TensorFlow.js – versión educativa interactiva
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
  const [categories, setCategories] = useState(['Gato', 'Casa', 'Sol']);
  const [currentCat, setCurrentCat] = useState('Gato');
  const [dataset, setDataset] = useState({});
  const [model, setModel] = useState(null);
  const [activationModel, setActivationModel] = useState(null);
  const [training, setTraining] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [activations, setActivations] = useState([]);
  const [currentEpoch, setCurrentEpoch] = useState(null);
  const [activeLayerIndex, setActiveLayerIndex] = useState(-1);
  const [skipEpochs, setSkipEpochs] = useState(false);
  const skipEpochsRef = useRef(false);

  useEffect(() => {
    const saved = localStorage.getItem('clasificador_dataset_v1');
    if (saved) setDataset(JSON.parse(saved));
  }, []);

  useEffect(() => {
    localStorage.setItem('clasificador_dataset_v1', JSON.stringify(dataset));
  }, [dataset]);

  const capture = () => {
    if (!webcamRef.current) return alert('Cámara no lista');
    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return alert('No se pudo capturar la imagen');
    setDataset((prev) => {
      const copy = { ...prev };
      if (!copy[currentCat]) copy[currentCat] = [];
      copy[currentCat].push(imageSrc);
      return copy;
    });
  };

  const addCategory = () => {
    const name = prompt('Nombre de la nueva categoría:');
    if (!name) return;
    setCategories((c) => (c.includes(name) ? c : [...c, name]));
    setCurrentCat(name);
  };

  const clearDataset = () => {
    if (!confirm('¿Borrar todo el dataset?')) return;
    setDataset({});
  };

  const loadImage = (dataUrl) =>
    new Promise((resolve) => {
      const img = new Image();
      img.src = dataUrl;
      img.onload = () => resolve(img);
    });

  const buildTensorsFromDataset = async () => {
    const cats = Object.keys(dataset).filter((k) => dataset[k].length > 0);
    if (cats.length < 2) throw new Error('Se necesitan al menos 2 categorías con imágenes.');
    const xs = [];
    const ys = [];
    for (let i = 0; i < cats.length; i++) {
      const cat = cats[i];
      for (const b64 of dataset[cat]) {
        const img = await loadImage(b64);
        const tensor = tf.browser
          .fromPixels(img)
          .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE])
          .toFloat()
          .div(255.0);
        xs.push(tensor);
        ys.push(i);
      }
    }
    const X = tf.stack(xs);
    const Y = tf.tensor1d(ys, 'int32');
    const Y_onehot = tf.oneHot(Y, cats.length);
    return { X, Y: Y_onehot, labels: cats };
  };

  const createModel = (numClasses) => {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({ inputShape: [IMAGE_SIZE, IMAGE_SIZE, CHANNELS], filters: 16, kernelSize: 3, activation: 'relu' }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu' }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));
    model.compile({ optimizer: tf.train.adam(0.001), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
    return model;
  };

  const waitOrSkip = async (ms) => {
    const step = 50;
    let waited = 0;
    while (waited < ms) {
      if (skipEpochsRef.current) break;
      await new Promise((r) => setTimeout(r, step));
      waited += step;
    }
  };

  const handleSkipEpochs = () => {
    if (!training) return;
    skipEpochsRef.current = true;
    setSkipEpochs(true);
  };

  const trainModel = async () => {
    setTraining(true);
    setCurrentEpoch(null);
    setSkipEpochs(false);
    skipEpochsRef.current = false;
    setActiveLayerIndex(-1);

    try {
      const { X, Y, labels } = await buildTensorsFromDataset();
      const m = createModel(labels.length);
      setModel(m);
      const activationM = tf.model({ inputs: m.input, outputs: m.layers.map((l) => l.output) });
      setActivationModel(activationM);

      const layerCount = 6;
      const perLayerDelay = 900;

      await m.fit(X, Y, {
        epochs: EPOCHS,
        batchSize: BATCH_SIZE,
        shuffle: true,
        callbacks: {
          onEpochEnd: async (epoch) => {
            setCurrentEpoch(epoch + 1);
            const sample = X.slice([0, 0, 0, 0], [1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS]);
            const acts = activationM.predict(sample);
            const arr = await Promise.all(
              acts.map(async (a) => {
                const mean = (await a.mean().data())[0];
                a.dispose();
                return mean;
              })
            );
            setActivations(arr);

            // animación capa por capa
            for (let layerIdx = 0; layerIdx < layerCount; layerIdx++) {
              if (skipEpochsRef.current) break;
              setActiveLayerIndex(layerIdx);
              await waitOrSkip(perLayerDelay);
            }
            setActiveLayerIndex(-1);

            if (skipEpochsRef.current) {
              skipEpochsRef.current = false;
              setSkipEpochs(false);
            }
            await tf.nextFrame();
          },
        },
      });

      X.dispose();
      Y.dispose();
      setCurrentEpoch(null);
      alert('🎉 Entrenamiento completado');
    } catch (err) {
      alert('Error durante el entrenamiento: ' + err.message);
    } finally {
      setTraining(false);
      skipEpochsRef.current = false;
      setSkipEpochs(false);
      setActiveLayerIndex(-1);
    }
  };

  const predictOnce = async () => {
    if (!model || !activationModel) return alert('Entrena el modelo primero');
    const imageSrc = webcamRef.current.getScreenshot();
    const img = await loadImage(imageSrc);
    const tensor = tf.browser
      .fromPixels(img)
      .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE])
      .toFloat()
      .div(255.0)
      .expandDims(0);

    const acts = activationModel.predict(tensor);
    const arrActs = await Promise.all(
      acts.map(async (a) => {
        const mean = (await a.mean().data())[0];
        a.dispose();
        return mean;
      })
    );
    setActivations(arrActs);

    const perLayerDelay = 700;
    const layerCount = 5;
    for (let i = 0; i < layerCount; i++) {
      setActiveLayerIndex(i);
      await new Promise((r) => setTimeout(r, perLayerDelay));
    }
    setActiveLayerIndex(-1);

    const preds = model.predict(tensor);
    const arr = await preds.data();
    const labels = Object.keys(dataset).filter((k) => dataset[k].length > 0);
    const pairs = labels.map((l, i) => ({ label: l, prob: arr[i] }));
    pairs.sort((a, b) => b.prob - a.prob);
    setPrediction(pairs);

    tensor.dispose();
    preds.dispose();
  };

  const exportDataset = () => {
    const blob = new Blob([JSON.stringify(dataset)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'dataset_clasificador.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const importDataset = (ev) => {
    const f = ev.target.files[0];
    if (!f) return;
    const r = new FileReader();
    r.onload = (e) => {
      try {
        const obj = JSON.parse(e.target.result);
        setDataset(obj);
        alert('Dataset importado');
      } catch {
        alert('Error leyendo archivo');
      }
    };
    r.readAsText(f);
  };

  // --- Visualización de la red ---
  const NetworkViz = ({ activations = [], categories = [], prediction = [], training, currentEpoch, totalEpochs, activeLayerIndex }) => {
    const safeCategories = Array.isArray(categories) ? categories : [];
    const safePrediction = Array.isArray(prediction) ? prediction : [];
    const layerNames = ['Conv1', 'Pool1', 'Conv2', 'Pool2', 'Densa'];
    const normalize = (x) => Math.min(1, Math.max(0, x / 0.5));
    const colorFromActivation = (a, active) => {
      if (!active) return 'rgba(180,180,180,0.4)';
      const n = normalize(a);
      const r = Math.round(220 + 35 * n);
      const g = Math.round(180 - 120 * n);
      return `rgba(${r},${g},50,0.9)`;
    };
    const topPrediction =
      safePrediction.length > 0
        ? safePrediction.reduce((max, p) => (p.prob > max.prob ? p : max), safePrediction[0])
        : null;

    return (
      <div className="w-full border rounded mt-3 relative flex flex-col items-center overflow-x-auto">
        <h3 className="text-sm font-semibold my-2 text-center">Visualización de la red neuronal</h3>

        {training && currentEpoch != null && (
          <div
            style={{
              position: 'absolute',
              top: 15,
              right: 20,
              background: 'rgba(255,255,255,0.9)',
              borderRadius: '12px',
              padding: '10px 18px',
              fontSize: '1.3rem',
              fontWeight: 'bold',
              boxShadow: '0 0 8px rgba(0,0,0,0.3)',
            }}
          >
            🧠 Época {currentEpoch} / {totalEpochs}
          </div>
        )}

        <div style={{ display: 'flex', flexDirection: 'row', gap: '60px', minWidth: '1100px', padding: '10px' }}>
          {layerNames.map((name, i) => {
            const a = normalize(activations[i] || 0);
            const active = i === activeLayerIndex;
            const color = colorFromActivation(a, active);
            if (name.startsWith('Conv')) {
              const filters = 12;
              return (
                <div key={name} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <div className="text-xs mb-1 font-semibold">{name}</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', width: '70px', gap: '2px' }}>
                    {Array.from({ length: filters }).map((_, j) => (
                      <div
                        key={j}
                        style={{
                          width: '18px',
                          height: '18px',
                          borderRadius: '3px',
                          backgroundColor: color,
                          boxShadow: active ? '0 0 10px 3px rgba(255,150,0,0.8)' : 'none',
                          transition: 'background-color 2s ease-in-out, box-shadow 2s ease-in-out',
                        }}
                      />
                    ))}
                  </div>
                </div>
              );
            }
            if (name.startsWith('Pool')) {
              return (
                <div key={name} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <div className="text-xs mb-1 font-semibold">{name}</div>
                  <div
                    style={{
                      width: '20px',
                      height: active ? '130px' : '50px',
                      backgroundColor: color,
                      borderRadius: '5px',
                      boxShadow: active ? '0 0 12px 3px rgba(255,150,0,0.8)' : 'none',
                      transition: 'height 2s ease-in-out, background-color 2s ease-in-out, box-shadow 2s ease-in-out',
                    }}
                  />
                </div>
              );
            }
            if (name === 'Densa') {
              return (
                <div key={name} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <div className="text-xs mb-1 font-semibold">{name}</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {Array.from({ length: 6 }).map((_, j) => (
                      <div
                        key={j}
                        style={{
                          width: '24px',
                          height: '24px',
                          borderRadius: '50%',
                          backgroundColor: color,
                          boxShadow: active ? '0 0 10px 3px rgba(255,150,0,0.8)' : 'none',
                          transition: 'background-color 2s ease-in-out, box-shadow 2s ease-in-out',
                        }}
                      />
                    ))}
                  </div>
                </div>
              );
            }
            return null;
          })}

          {/* Capa de salida */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div className="text-xs mb-1 font-semibold">Categorías</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {safeCategories.map((cat, i) => {
                const isPredicted = topPrediction && topPrediction.label === cat;
                return (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <div
                      style={{
                        width: 30,
                        height: 30,
                        borderRadius: '50%',
                        border: '1px solid #333',
                        backgroundColor: isPredicted ? 'rgba(255, 40, 40, 0.95)' : 'rgba(60, 200, 60, 0.9)',
                        boxShadow: isPredicted ? '0 0 14px 4px rgba(255,80,80,0.8)' : 'none',
                        transition: 'all 0.8s ease-in-out',
                      }}
                    />
                    <span className="text-xs">{cat}</span>
                  </div>
                );
              })}
            </div>
          </div>
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
          <Webcam audio={false} screenshotFormat="image/jpeg" ref={webcamRef} videoConstraints={{ facingMode: 'user' }} width={320} height={240} />
          <div className="mt-2 flex gap-2">
            <select value={currentCat} onChange={(e) => setCurrentCat(e.target.value)}>
              {categories.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
            <button onClick={capture} className="px-2 py-1 border rounded">
              Capturar
            </button>
            <button onClick={addCategory} className="px-2 py-1 border rounded">
              Nueva categoría
            </button>
          </div>

          <div className="mt-3">
            <strong>Dataset (por categoría)</strong>
            <div className="max-h-40 overflow-auto border p-2 mt-1">
              {Object.keys(dataset).length === 0 && <div className="text-sm text-gray-500">(vacío)</div>}
              {Object.entries(dataset).map(([k, arr]) => (
                <div key={k} className="mb-2">
                  <div className="font-semibold">
                    {k} — {arr.length} imágenes
                  </div>
                  <div className="flex gap-1 overflow-auto">
                    {arr.slice(-6).map((b64, i) => (
                      <img key={i} src={b64} style={{ width: 50, height: 50, objectFit: 'cover' }} alt="m" />
                    ))}
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-2 flex gap-2">
              <button onClick={trainModel} disabled={training} className="px-3 py-1 border rounded">
                Entrenar
              </button>
              <button onClick={handleSkipEpochs} disabled={!training} className="px-3 py-1 border rounded">
                ⏩ Adelantar épocas
              </button>
              <button onClick={predictOnce} className="px-3 py-1 border rounded">
                Probar (predecir)
              </button>
              <button onClick={exportDataset} className="px-3 py-1 border rounded">
                Exportar dataset
              </button>
              <label className="px-3 py-1 border rounded cursor-pointer">
                Importar
                <input type="file" accept="application/json" onChange={importDataset} style={{ display: 'none' }} />
              </label>
              <button onClick={clearDataset} className="px-3 py-1 border rounded">
                Borrar dataset
              </button>
            </div>
          </div>
        </div>

        <div>
          <NetworkViz
            activations={activations}
            categories={categories}
            prediction={prediction}
            training={training}
            currentEpoch={currentEpoch}
            totalEpochs={EPOCHS}
            activeLayerIndex={activeLayerIndex}
          />

          <div className="mt-3">
            <strong>Predicción</strong>
            <div className="border p-2 mt-1">
              {!prediction && <div className="text-sm text-gray-500">Realiza una predicción para ver los porcentajes.</div>}
              {prediction && (
                <div>
                  {prediction.map((p, i) => (
                    <div key={i}>
                      {p.label}: {(p.prob * 100).toFixed(1)}%
                    </div>
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

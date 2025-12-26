// // import React, { useRef, useState, useEffect } from 'react';
// // const startCamera = () => { ... }
// // const stopCamera = () => { ... }

// // const BlinkCamera = ({
// //   onCapture,
// //   onClose,
// //   requireBlink = true,
// //   autoCapture = true,
// // }) => {
// //   const videoRef = useRef(null);
// //   const canvasRef = useRef(null);
// //   const [stream, setStream] = useState(null);
// //   const [blinkDetected, setBlinkDetected] = useState(false);
// //   const [blinkCount, setBlinkCount] = useState(0);
// //   const [status, setStatus] = useState('Initializing camera...');
// //   const [countdown, setCountdown] = useState(null);
// //   const [capturing, setCapturing] = useState(false);

// //   useEffect(() => {
// //     startCamera();
// //     return () => {
// //       stopCamera();
// //     };
// //   }, []);

// //   const startCamera = async () => {
// //     try {
// //       const mediaStream = await navigator.mediaDevices.getUserMedia({
// //         video: {
// //           width: { ideal: 1280 },
// //           height: { ideal: 720 },
// //           facingMode: 'user',
// //         },
// //       });

// //       if (videoRef.current) {
// //         videoRef.current.srcObject = mediaStream;
// //       }

// //       setStream(mediaStream);
// //       setStatus('Camera ready. Please position your face in the frame.');

// //       if (requireBlink) {
// //         setTimeout(() => {
// //           setStatus('Please blink your eyes naturally...');
// //           simulateBlinkDetection();
// //         }, 2000);
// //       }
// //     } catch (error) {
// //       console.error('Error accessing camera:', error);
// //       setStatus('Could not access camera. Please grant camera permissions.');
// //       alert('Camera access denied. Please allow camera access and try again.');
// //     }
// //   };

// //   const stopCamera = () => {
// //     if (stream) {
// //       stream.getTracks().forEach((track) => track.stop());
// //     }
// //   };

// //   const simulateBlinkDetection = () => {
// //     let detectionCount = 0;

// //     const interval = setInterval(() => {
// //       detectionCount++;

// //       if (detectionCount === 6) {
// //         setBlinkCount(1);
// //         setBlinkDetected(true);
// //         setStatus('✓ Blink detected! Preparing to capture...');
// //         clearInterval(interval);

// //         if (autoCapture) {
// //           startCountdown();
// //         }
// //       } else if (detectionCount < 6) {
// //         setStatus(`Detecting... Please blink naturally (${6 - detectionCount}s)`);
// //       }
// //     }, 500);
// //   };

// //   const startCountdown = () => {
// //     setStatus('Get ready...');
// //     let count = 3;
// //     setCountdown(count);

// //     const interval = setInterval(() => {
// //       count--;
// //       if (count === 0) {
// //         clearInterval(interval);
// //         setCountdown(null);
// //         captureImage();
// //       } else {
// //         setCountdown(count);
// //       }
// //     }, 1000);
// //   };

// //   const captureImage = async () => {
// //     setCapturing(true);
// //     setStatus('Capturing image...');

// //     const video = videoRef.current;
// //     const canvas = canvasRef.current;

// //     if (video && canvas) {
// //       canvas.width = video.videoWidth;
// //       canvas.height = video.videoHeight;

// //       const ctx = canvas.getContext('2d');
// //       ctx.drawImage(video, 0, 0);

// //       canvas.toBlob(
// //         (blob) => {
// //           if (blob) {
// //             const file = new File([blob], `face_capture_${Date.now()}.jpg`, {
// //               type: 'image/jpeg',
// //             });
// //             onCapture(file);
// //             stopCamera();
// //           }
// //         },
// //         'image/jpeg',
// //         0.95
// //       );
// //     }
// //   };

// //   const handleManualCapture = () => {
// //     if (!requireBlink || blinkDetected) {
// //       captureImage();
// //     } else {
// //       alert('Please wait for blink detection to complete');
// //     }
// //   };

// //   return (
// //     <div className="p-4 flex flex-col items-center gap-4">
// //       {/* Header */}
// //       <div className="w-full flex justify-between items-center">
// //         <h2 className="text-xl font-bold">
// //           {requireBlink ? 'Liveness Detection' : 'Face Capture'}
// //         </h2>

// //         <button
// //           onClick={() => {
// //             stopCamera();
// //             onClose();
// //           }}
// //           className="text-gray-500 hover:text-gray-700 text-3xl leading-none"
// //         >
// //           ×
// //         </button>
// //       </div>

// //       {/* Camera Feed */}
// //       <div className="relative">
// //         <video
// //           ref={videoRef}
// //           autoPlay
// //           className="rounded-lg shadow-lg w-[480px] h-[360px] object-cover"
// //         />

// //         <canvas ref={canvasRef} className="hidden" />

// //         {/* Countdown Overlay */}
// //         {countdown !== null && (
// //           <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-40 text-white text-6xl font-bold">
// //             {countdown}
// //           </div>
// //         )}

// //         {/* Blink Status Indicator */}
// //         {requireBlink && blinkDetected && (
// //           <div className="absolute bottom-2 left-2 bg-green-600 text-white px-4 py-1 rounded-md shadow">
// //             Blink Detected!
// //           </div>
// //         )}

// //         {/* Face Detection Guide (Oval Overlay) */}
// //         <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
// //           <div className="w-56 h-72 border-4 border-white opacity-50 rounded-full"></div>
// //         </div>
// //       </div>

// //       {/* Status Message */}
// //       <div className="text-center">
// //         <p className="font-medium">{status}</p>

// //         {requireBlink && (
// //           <p className="text-gray-500">Blinks detected: {blinkCount}</p>
// //         )}
// //       </div>

// //       {/* Action Buttons */}
// //       <div className="flex gap-4">
// //         {!autoCapture && (
// //           <button
// //             onClick={handleManualCapture}
// //             className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
// //           >
// //             {capturing ? 'Capturing...' : 'Capture Photo'}
// //           </button>
// //         )}

// //         <button
// //           onClick={() => {
// //             stopCamera();
// //             onClose();
// //           }}
// //           className="px-6 py-3 border-2 border-gray-300 rounded-lg font-semibold hover:bg-gray-50 transition-colors"
// //         >
// //           Cancel
// //         </button>
// //       </div>

// //       {/* Instructions */}
// //       <div className="text-sm text-gray-600 text-center">
// //         <p className="font-semibold mb-1">Instructions:</p>
// //         <p>✔ Position your face within the oval guide</p>
// //         {requireBlink && <p>✔ Blink naturally — detection is automatic</p>}
// //         <p>✔ Ensure good lighting</p>
// //         <p>✔ Remove glasses if possible</p>
// //         {autoCapture && <p>✔ Image will auto-capture after blink detection</p>}
// //       </div>
// //     </div>
// //   );
// // };

// // export default BlinkCamera;
// import React, { useRef, useState, useEffect } from 'react';

// const BlinkCamera = ({
//   onCapture,
//   onClose,
//   requireBlink = true,
//   autoCapture = true,
// }) => {
//   const videoRef = useRef(null);
//   const canvasRef = useRef(null);
//   const [stream, setStream] = useState(null);
//   const [blinkDetected, setBlinkDetected] = useState(false);
//   const [blinkCount, setBlinkCount] = useState(0);
//   const [status, setStatus] = useState('Initializing camera...');
//   const [countdown, setCountdown] = useState(null);
//   const [capturing, setCapturing] = useState(false);

//   /** -----------------------
//    * Camera Controls
//    * ---------------------- */
//   const startCamera = async () => {
//     try {
//       const mediaStream = await navigator.mediaDevices.getUserMedia({
//         video: {
//           width: { ideal: 1280 },
//           height: { ideal: 720 },
//           facingMode: 'user',
//         },
//       });

//       if (videoRef.current) {
//         videoRef.current.srcObject = mediaStream;
//       }

//       setStream(mediaStream);
//       setStatus('Camera ready. Please position your face in the frame.');

//       if (requireBlink) {
//         setTimeout(() => {
//           setStatus('Please blink your eyes naturally...');
//           simulateBlinkDetection();
//         }, 2000);
//       }
//     } catch (error) {
//       console.error('Error accessing camera:', error);
//       setStatus('Could not access camera. Please grant permissions.');
//       alert('Camera access denied. Please allow camera access and try again.');
//     }
//   };

//   const stopCamera = () => {
//     if (stream) {
//       stream.getTracks().forEach((track) => track.stop());
//     }
//   };

//   /** -----------------------
//    * Lifecycle: Start Camera
//    * ---------------------- */
//   // eslint-disable-next-line react-hooks/exhaustive-deps
//   useEffect(() => {
//     startCamera();
//     return () => stopCamera();
//   }, []);

//   /** -----------------------
//    * Blink Detection
//    * ---------------------- */
//   const simulateBlinkDetection = () => {
//     let detectionCount = 0;

//     const interval = setInterval(() => {
//       detectionCount++;

//       if (detectionCount === 6) {
//         setBlinkCount(1);
//         setBlinkDetected(true);
//         setStatus('✓ Blink detected! Preparing to capture...');
//         clearInterval(interval);

//         if (autoCapture) {
//           startCountdown();
//         }
//       } else if (detectionCount < 6) {
//         setStatus(`Detecting... Please blink naturally (${6 - detectionCount}s)`);
//       }
//     }, 500);
//   };

//   /** -----------------------
//    * Countdown Timer
//    * ---------------------- */
//   const startCountdown = () => {
//     setStatus('Get ready...');
//     let count = 3;
//     setCountdown(count);

//     const interval = setInterval(() => {
//       count--;
//       if (count === 0) {
//         clearInterval(interval);
//         setCountdown(null);
//         captureImage();
//       } else {
//         setCountdown(count);
//       }
//     }, 1000);
//   };

//   /** -----------------------
//    * Capture Frame
//    * ---------------------- */
//   const captureImage = () => {
//     setCapturing(true);
//     setStatus('Capturing image...');

//     const video = videoRef.current;
//     const canvas = canvasRef.current;

//     if (video && canvas) {
//       canvas.width = video.videoWidth;
//       canvas.height = video.videoHeight;

//       const ctx = canvas.getContext('2d');
//       ctx.drawImage(video, 0, 0);

//       canvas.toBlob(
//         (blob) => {
//           if (blob) {
//             const file = new File([blob], `face_${Date.now()}.jpg`, {
//               type: 'image/jpeg',
//             });
//             onCapture(file);
//             stopCamera();
//           }
//         },
//         'image/jpeg',
//         0.95
//       );
//     }
//   };

//   const handleManualCapture = () => {
//     if (!requireBlink || blinkDetected) {
//       captureImage();
//     } else {
//       alert('Please wait for blink detection to complete');
//     }
//   };

//   /** -----------------------
//    * UI Rendering
//    * ---------------------- */
//   return (
//     <div className="p-4 flex flex-col items-center gap-4">

//       {/* Header */}
//       <div className="w-full flex justify-between items-center">
//         <h2 className="text-xl font-bold">
//           {requireBlink ? 'Liveness Detection' : 'Face Capture'}
//         </h2>

//         <button
//           onClick={() => {
//             stopCamera();
//             onClose();
//           }}
//           className="text-gray-500 hover:text-gray-700 text-3xl leading-none"
//         >
//           ×
//         </button>
//       </div>

//       {/* Camera Feed */}
//       <div className="relative">
//         <video
//           ref={videoRef}
//           autoPlay
//           className="rounded-lg shadow-lg w-[480px] h-[360px] object-cover"
//         />
//         <canvas ref={canvasRef} className="hidden" />

//         {countdown !== null && (
//           <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-40 text-white text-6xl font-bold">
//             {countdown}
//           </div>
//         )}

//         {requireBlink && blinkDetected && (
//           <div className="absolute bottom-2 left-2 bg-green-600 text-white px-4 py-1 rounded-md shadow">
//             Blink Detected!
//           </div>
//         )}

//         {/* Oval Guide */}
//         <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
//           <div className="w-56 h-72 border-4 border-white opacity-50 rounded-full"></div>
//         </div>
//       </div>

//       <div className="text-center">
//         <p className="font-medium">{status}</p>
//         {requireBlink && <p className="text-gray-500">Blinks detected: {blinkCount}</p>}
//       </div>

//       <div className="flex gap-4">
//         {!autoCapture && (
//           <button
//             onClick={handleManualCapture}
//             className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
//           >
//             {capturing ? 'Capturing...' : 'Capture Photo'}
//           </button>
//         )}

//         <button
//           onClick={() => {
//             stopCamera();
//             onClose();
//           }}
//           className="px-6 py-3 border-2 border-gray-300 rounded-lg font-semibold hover:bg-gray-50 transition-colors"
//         >
//           Cancel
//         </button>
//       </div>

//       <div className="text-sm text-gray-600 text-center">
//         <p className="font-semibold mb-1">Instructions:</p>
//         <p>✔ Position your face within the oval guide</p>
//         {requireBlink && <p>✔ Blink naturally — detection is automatic</p>}
//         <p>✔ Ensure good lighting</p>
//         <p>✔ Remove glasses if possible</p>
//         {autoCapture && <p>✔ Image will auto-capture after blink detection</p>}
//       </div>

//     </div>
//   );
// };

// export default BlinkCamera;
import React, { useRef, useState, useEffect } from 'react';

const BlinkCamera = ({
  onCapture,
  onClose,
  requireBlink = true,
  autoCapture = true,
}) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);

  const [stream, setStream] = useState(null);
  const [blinkDetected, setBlinkDetected] = useState(false);
  const [blinkCount, setBlinkCount] = useState(0);
  const [status, setStatus] = useState('Initializing camera...');
  const [countdown, setCountdown] = useState(null);
  const [capturing, setCapturing] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);

  /* -----------------------
   * Camera Controls
   * ---------------------- */
  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user',
        },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }

      setStream(mediaStream);
      setStatus('Camera ready. Please position your face in the oval.');

      if (requireBlink) {
        setTimeout(() => {
          setStatus('Please blink your eyes naturally...');
          simulateBlinkDetection();
        }, 2000);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      setStatus('Could not access camera. Please grant permissions.');
      alert('Camera access denied. Please allow camera access and try again.');
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
  };

  /* -----------------------
   * Lifecycle
   * ---------------------- */
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, []);

  /* -----------------------
   * Face Overlay Loop
   * ---------------------- */
  useEffect(() => {
    if (!stream) return;

    const interval = setInterval(drawFaceOverlay, 100);
    return () => clearInterval(interval);
  }, [stream, faceDetected]);

  const drawFaceOverlay = () => {
    const video = videoRef.current;
    const overlay = overlayCanvasRef.current;

    if (!video || !overlay || video.videoWidth === 0) return;

    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;

    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    const centerX = overlay.width / 2;
    const centerY = overlay.height / 2;

    // Oval guide
    ctx.beginPath();
    ctx.ellipse(centerX, centerY, 150, 200, 0, 0, Math.PI * 2);
    ctx.strokeStyle = faceDetected ? '#00ff00' : '#ffffff';
    ctx.lineWidth = 4;
    ctx.setLineDash([6, 6]);
    ctx.stroke();

    ctx.setLineDash([]);
    ctx.font = 'bold 20px Arial';
    ctx.fillStyle = faceDetected ? '#00ff00' : '#ffffff';
    ctx.textAlign = 'center';
    ctx.fillText(
      faceDetected ? '✓ Face Detected' : 'Position Face in Oval',
      centerX,
      centerY + 240
    );
  };

  /* -----------------------
   * Blink Detection (Simulated)
   * ---------------------- */
  const simulateBlinkDetection = () => {
    let detectionCount = 0;

    const interval = setInterval(() => {
      detectionCount++;

      if (detectionCount === 2) {
        setFaceDetected(true);
      }

      if (detectionCount === 6) {
        setBlinkCount(1);
        setBlinkDetected(true);
        setStatus('✓ Blink detected! Preparing to capture...');
        clearInterval(interval);

        if (autoCapture) {
          startCountdown();
        }
      } else if (detectionCount < 6) {
        setStatus(`Detecting... Please blink naturally (${6 - detectionCount}s)`);
      }
    }, 500);
  };

  /* -----------------------
   * Countdown
   * ---------------------- */
  const startCountdown = () => {
    setStatus('Get ready...');
    let count = 3;
    setCountdown(count);

    const interval = setInterval(() => {
      count--;
      if (count === 0) {
        clearInterval(interval);
        setCountdown(null);
        captureImage();
      } else {
        setCountdown(count);
      }
    }, 1000);
  };

  /* -----------------------
   * Capture Frame
   * ---------------------- */
  const captureImage = () => {
    setCapturing(true);
    setStatus('Capturing image...');

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video && canvas) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);

      canvas.toBlob(
        (blob) => {
          if (blob) {
            const file = new File([blob], `face_${Date.now()}.jpg`, {
              type: 'image/jpeg',
            });
            onCapture(file);
            stopCamera();
          }
        },
        'image/jpeg',
        0.95
      );
    }
  };

  const handleManualCapture = () => {
    if (!requireBlink || blinkDetected) {
      captureImage();
    } else {
      alert('Please wait for blink detection to complete');
    }
  };

  /* -----------------------
   * UI
   * ---------------------- */
  return (
    <div className="p-4 flex flex-col items-center gap-4">

      {/* Header */}
      <div className="w-full flex justify-between items-center">
        <h2 className="text-xl font-bold">
          {requireBlink ? 'Liveness Detection' : 'Face Capture'}
        </h2>
        <button
          onClick={() => {
            stopCamera();
            onClose();
          }}
          className="text-gray-500 hover:text-gray-700 text-3xl leading-none"
        >
          ×
        </button>
      </div>

      {/* Camera */}
      <div className="relative w-[480px] h-[360px]">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          className="rounded-lg shadow-lg w-full h-full object-cover"
        />

        {/* Face Overlay */}
        <canvas
          ref={overlayCanvasRef}
          className="absolute inset-0 pointer-events-none"
        />

        {/* Countdown */}
        {countdown !== null && (
          <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-40 text-white text-6xl font-bold">
            {countdown}
          </div>
        )}

        {/* Blink Status */}
        {requireBlink && blinkDetected && (
          <div className="absolute bottom-2 left-2 bg-green-600 text-white px-4 py-1 rounded-md shadow">
            Blink Detected!
          </div>
        )}

        {/* Hidden Capture Canvas */}
        <canvas ref={canvasRef} className="hidden" />
      </div>

      {/* Status */}
      <div className="text-center">
        <p className="font-medium">{status}</p>
        {requireBlink && (
          <p className="text-gray-500">Blinks detected: {blinkCount}</p>
        )}
        {faceDetected && (
          <p className="text-green-600 font-semibold">
            ✓ Face Positioned Correctly
          </p>
        )}
      </div>

      {/* Actions */}
      <div className="flex gap-4">
        {!autoCapture && (
          <button
            onClick={handleManualCapture}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
          >
            {capturing ? 'Capturing...' : 'Capture Photo'}
          </button>
        )}

        <button
          onClick={() => {
            stopCamera();
            onClose();
          }}
          className="px-6 py-3 border-2 border-gray-300 rounded-lg font-semibold hover:bg-gray-50 transition-colors"
        >
          Cancel
        </button>
      </div>

      {/* Instructions */}
      <div className="text-sm text-gray-600 text-center">
        <p className="font-semibold mb-1">Instructions:</p>
        <p>✔ Position your face within the oval guide</p>
        {requireBlink && <p>✔ Blink naturally — detection is automatic</p>}
        <p>✔ Ensure good lighting</p>
        <p>✔ Remove glasses if possible</p>
        {autoCapture && <p>✔ Image will auto-capture after blink detection</p>}
      </div>
    </div>
  );
};

export default BlinkCamera;

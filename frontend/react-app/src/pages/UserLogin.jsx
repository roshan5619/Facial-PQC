import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import BlinkCamera from '../components/BlinkCamera';
import { authAPI } from '../services/api';

const UserLogin = () => {
  const navigate = useNavigate();
  const [showCamera, setShowCamera] = useState(false);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);

  const handleCapture = async (capturedImage) => {
    setLoading(true);
    setMessage(null);
    setShowCamera(false);

    try {
      const response = await authAPI.login(capturedImage);
      const {
        success,
        access_token,
        username,
        // similarity_score,
        message: msg,
      } = response.data;

      if (success) {
        // Store token
        localStorage.setItem('access_token', access_token);
        localStorage.setItem('username', username);

        setMessage({
          type: 'success',
          text: `✓ Authentication successful! Welcome ${username}`,
        });

        setTimeout(() => {
          navigate('/dashboard');
        }, 1500);
      } else {
        setMessage({
          type: 'error',
          text: msg || 'Authentication failed',
        });
      }
    } catch (error) {
      setMessage({
        type: 'error',
        text:
          error.response?.data?.detail ||
          'Authentication failed. Please try again.',
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6 bg-gray-100">
      
      {/* Header */}
      <div className="text-center mb-8">
        <div className="text-6xl mb-3">🔓</div>
        <h1 className="text-3xl font-bold">Face Login</h1>
        <p className="text-gray-600 mt-1">Authenticate with face recognition</p>
      </div>

      {/* Message Display */}
      {message && (
        <div
          className={`mb-6 p-4 rounded-lg w-full max-w-md ${
            message.type === 'success'
              ? 'bg-green-100 text-green-800 border border-green-300'
              : 'bg-red-100 text-red-800 border border-red-300'
          }`}
        >
          {message.text}
        </div>
      )}

      {/* Login Button */}
      <button
        onClick={() => setShowCamera(true)}
        disabled={loading}
        className="w-full max-w-md bg-gradient-to-r from-blue-600 to-purple-600 
                   text-white px-6 py-4 rounded-xl font-semibold text-lg 
                   hover:from-blue-700 hover:to-purple-700 
                   disabled:opacity-50 disabled:cursor-not-allowed 
                   transition-all transform hover:scale-105 shadow-lg mb-6"
      >
        {loading ? (
          <div className="flex items-center justify-center gap-2">
            <span className="loader"></span>
            Authenticating...
          </div>
        ) : (
          <span>📷 Authenticate with Face</span>
        )}
      </button>

      {/* Instructions */}
      <div className="w-full max-w-md text-left text-gray-700 bg-white p-6 rounded-xl shadow">
        <h2 className="font-semibold text-lg mb-3">How it works:</h2>
        <ul className="list-disc ml-5 space-y-1 text-sm">
          <li>Click the button above</li>
          <li>Allow camera access when prompted</li>
          <li>Look at the camera and blink naturally</li>
          <li>Image will be captured automatically</li>
          <li>System will verify your identity</li>
        </ul>
      </div>

      {/* Back Button */}
      <button
        onClick={() => navigate('/')}
        className="text-blue-600 hover:text-blue-800 text-sm font-medium mt-6"
      >
        ← Back to Home
      </button>

      {/* Blink Camera Modal */}
      {showCamera && (
        <BlinkCamera
          onCapture={handleCapture}
          onClose={() => setShowCamera(false)}
          requireBlink={true}
          autoCapture={true}
        />
      )}
    </div>
  );
};

export default UserLogin;

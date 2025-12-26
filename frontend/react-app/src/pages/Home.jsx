import React from 'react';

import { useNavigate } from 'react-router-dom';

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-6">
      
      {/* Header */}
      <div className="text-center mb-12">
        <div className="text-6xl mb-4">🔐</div>
        <h1 className="text-4xl font-bold">Face Authentication System</h1>
        <p className="text-gray-600 mt-2">
          Secure access with AI-powered face recognition
        </p>
      </div>

      {/* Action Cards */}
      <div className="flex flex-col md:flex-row gap-10 mb-12">

        {/* Login Card */}
        <button
          onClick={() => navigate('/login')}
          className="group bg-white rounded-2xl p-8 w-72 shadow-2xl hover:shadow-3xl 
                     transform hover:-translate-y-2 transition-all duration-300 text-center"
        >
          <div className="text-5xl mb-3">👤</div>

          <h2 className="text-2xl font-bold mb-2">LOGIN</h2>

          <p className="text-gray-500 mb-4">
            Already registered? Authenticate with your face
          </p>

          <span className="inline-block bg-blue-600 group-hover:bg-blue-700 text-white 
                           px-6 py-2 rounded-lg font-semibold">
            Access System
          </span>
        </button>

        {/* Register Card */}
        <button
          onClick={() => navigate('/register')}
          className="group bg-white rounded-2xl p-8 w-72 shadow-2xl hover:shadow-3xl 
                     transform hover:-translate-y-2 transition-all duration-300 text-center"
        >
          <div className="text-5xl mb-3">✍️</div>

          <h2 className="text-2xl font-bold mb-2">REGISTER</h2>

          <p className="text-gray-500 mb-4">
            New user? Create your account with face enrollment
          </p>

          <span className="inline-block bg-green-600 group-hover:bg-green-700 text-white 
                           px-6 py-2 rounded-lg font-semibold">
            Get Started
          </span>
        </button>
      </div>

      {/* Features */}
      <div className="text-center mb-12">
        <h3 className="font-semibold text-lg mb-4">System Features:</h3>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          <div className="flex flex-col items-center text-gray-700">
            <div className="text-3xl">⚡</div>
            <span>Instant Access</span>
          </div>

          <div className="flex flex-col items-center text-gray-700">
            <div className="text-3xl">🛡️</div>
            <span>Secure</span>
          </div>

          <div className="flex flex-col items-center text-gray-700">
            <div className="text-3xl">👁️</div>
            <span>Liveness Check</span>
          </div>

          <div className="flex flex-col items-center text-gray-700">
            <div className="text-3xl">🤖</div>
            <span>AI-Powered</span>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="text-gray-600 mt-10">
        Powered by Quantum ML & Advanced Face Recognition
      </div>
    </div>
  );
};

export default Home;

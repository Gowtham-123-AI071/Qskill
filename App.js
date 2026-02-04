// frontend/src/App.js
import React, { useState } from "react";

function App(){
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  async function onSubmit(e){
    e.preventDefault();
    setLoading(true);
    try {
      const resp = await fetch("https://https://spam.example.com", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({message: text})
      });
      const data = await resp.json();
      setResult(data);
    } catch(err){
      setResult({error: err.message});
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{maxWidth:720, margin:"40px auto", fontFamily:"Arial"}}>
      <h1>Spam Detector</h1>
      <p>Paste a message and see if it’s spam or not.</p>
      <form onSubmit={onSubmit}>
        <textarea rows="6" style={{width:"100%"}} value={text} onChange={e=>setText(e.target.value)} />
        <button type="submit" disabled={loading || !text.trim()} style={{marginTop:10}}>
          {loading ? "Checking..." : "Check"}
        </button>
      </form>

      {result && !result.error && (
        <div style={{marginTop:20, padding:20, border:"1px solid #eee"}}>
          <h3>Prediction: {result.prediction.toUpperCase()}</h3>
          <p>Confidence: {(result.confidence*100).toFixed(2)}%</p>
        </div>
      )}
      {result && result.error && <div style={{color:"red"}}>{result.error}</div>}
    </div>
  );
}

export default App;

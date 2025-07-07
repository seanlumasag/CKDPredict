import logo from "../assets/logo.png";

function PredictionForm({
  age,
  bp,
  sg,
  al,
  setAge,
  setBp,
  setSg,
  setAl,
  loading,
  isEditing,
  handleSubmit,
  result,
  error,
}) {
  return (
    <div className="prediction-form">
      <div className="hero">
        <div className="text-container">
          <h1>CKDPredict</h1>
          <p>
            Chronic Kidney Disease Risk Predictor, ML-Powered, Full-Stack App
          </p>
        </div>
        <div className="img-container">
          <img src={"logo"} alt="" />
        </div>
      </div>
      <div className="form-container">
        <form onSubmit={handleSubmit}>
          <label>
            Age:
            <input
              type="number"
              value={age}
              onChange={(e) => setAge(e.target.value)}
            />
          </label>
          <label>
            Blood Pressure:
            <input
              type="number"
              value={bp}
              onChange={(e) => setBp(e.target.value)}
            />
          </label>
          <label>
            Specific Gravity:
            <input
              type="number"
              step="0.01"
              value={sg}
              onChange={(e) => setSg(e.target.value)}
            />
          </label>
          <label>
            Albumin:
            <input
              type="number"
              value={al}
              onChange={(e) => setAl(e.target.value)}
            />
          </label>
          <button type="submit" disabled={loading}>
            {isEditing ? "Update Entry" : "Predict CKD Risk"}
          </button>
        </form>
      </div>

      <p>Result:<span className="result">{result}</span></p>
      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  );
}

export default PredictionForm;

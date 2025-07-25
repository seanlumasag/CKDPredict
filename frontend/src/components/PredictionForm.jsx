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
      {/* Top section with app title and logo */}
      <div className="hero">
        <div className="text-container">
          <h1>CKDPredict</h1>
          <p>- Risk Predictor for Chronic Kidney Disease</p>
          <p>- ML-Powered Full-Stack App</p>
          <p>- Loading may take up to a minute</p>
        </div>
        <div className="img-container">
          <img src={logo} alt="" />
        </div>
      </div>

      {/* Form section */}
      <div className="form-container">
        <form onSubmit={handleSubmit}>
          {/* Age input */}
          <label>
            Age:
            <input
              type="number"
              value={age}
              onChange={(e) => setAge(e.target.value)}
            />
          </label>

          {/* Blood pressure input */}
          <label>
            Blood Pressure:
            <input
              type="number"
              value={bp}
              onChange={(e) => setBp(e.target.value)}
            />
          </label>

          {/* Specific gravity input */}
          <label>
            Specific Gravity:
            <input
              type="number"
              step="0.01"
              value={sg}
              onChange={(e) => setSg(e.target.value)}
            />
          </label>

          {/* Albumin input */}
          <label>
            Albumin:
            <input
              type="number"
              value={al}
              onChange={(e) => setAl(e.target.value)}
            />
          </label>

          {/* Submit button (disabled when loading) */}
          <button type="submit" disabled={loading}>
            {isEditing ? "Update Entry" : "Predict CKD Risk"}
          </button>
        </form>
      </div>

      <p>
        Result: {"        "}
        {loading ? (
          <span className="result">Loading...</span>
        ) : error ? (
          <span
            className="result"
            style={{
              color: "white",
              marginLeft: "0.25rem",
              backgroundColor: "red",
            }}
          >
            {error}
          </span>
        ) : (
          <span className="result">{result}</span>
        )}
      </p>
    </div>
  );
}

export default PredictionForm;

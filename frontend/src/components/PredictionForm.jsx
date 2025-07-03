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
      <h1>CKDPredict</h1>
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
          {loading
            ? isEditing
              ? "Saving..."
              : "Predicting..."
            : isEditing
            ? "Update Entry"
            : "Predict CKD Risk"}
        </button>
      </form>
      {result && (
        <p>
          <strong>Result:</strong> {result}
        </p>
      )}
      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  );
}

export default PredictionForm;

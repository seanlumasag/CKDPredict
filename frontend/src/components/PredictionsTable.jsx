function PredictionsTable({
  history,
  loadingHistory,
  historyError,
  handleEdit,
  handleDelete,
}) {
  // If there's an error fetching history, show it in red
  if (historyError)
    return <p style={{ color: "red" }}>Error: {historyError}</p>;

  return (
    <div className="predictions-table">
      <h2>Past Predictions</h2>

      {/* Table displaying patient prediction history */}
      <table>
        <thead>
          <tr>
            <th>Age</th>
            <th>BP</th>
            <th>SG</th>
            <th>AL</th>
            <th>Prediction</th>
            <th>Date</th>
            <th>Actions</th>
          </tr>
        </thead>

        <tbody>
          {/* Loop through each prediction entry */}
          {history.map((patient) => (
            <tr key={patient.id}>
              {/* Display patient input values */}
              <td>{patient.age}</td>
              <td>{patient.bp}</td>
              <td>{patient.sg}</td>
              <td>{patient.al}</td>

              {/* Show prediction result as High or Low Risk */}
              <td>
                <div className="result-container">
                  <p className="result">
                    {patient.prediction ? "High Risk" : "Low Risk"}
                  </p>
                </div>
              </td>

              {/* Format and display date the entry was created */}
              <td>{new Date(patient.createdAt).toLocaleDateString()}</td>

              {/* Edit and Delete buttons for each entry */}
              <td>
                <div className="button-container">
                  <button onClick={() => handleEdit(patient)}>Edit</button>
                  <button onClick={() => handleDelete(patient.id)}>
                    Delete
                  </button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default PredictionsTable;

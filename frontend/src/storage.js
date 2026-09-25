const HISTORY_KEY = "analysisHistory";


// ============================================================
// GET HISTORY
// ============================================================

export function getHistory() {

  try {

    const saved =
      localStorage.getItem(
        HISTORY_KEY
      );

    return saved
      ? JSON.parse(saved)
      : [];

  } catch (error) {

    console.error(
      "Unable to read history:",
      error
    );

    return [];
  }
}


// ============================================================
// SAVE HISTORY
// ============================================================

export function saveHistory(records) {

  try {

    localStorage.setItem(
      HISTORY_KEY,
      JSON.stringify(records)
    );

  } catch (error) {

    console.error(
      "Unable to save history:",
      error
    );
  }
}
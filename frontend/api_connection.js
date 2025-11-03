const tbody = document.getElementById("crypto-body");
const pageNumber = document.getElementById("page-number");
const prevBtn = document.getElementById("prev-btn");
const nextBtn = document.getElementById("next-btn");

const modelSelect = document.getElementById("model-select"); // Assumes you add this <select> in HTML
const predictBtn = document.getElementById("predict-btn");
const result = document.getElementById("prediction-result");

let currentPage = 1;
const maxPage = 50; // CoinGecko's API limit

async function fetchData(page = 1) {
  try {
    const url = `https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=50&page=${page}&sparkline=true`;
    const response = await fetch(url);
    const data = await response.json();

    tbody.innerHTML = "";

    data.forEach((coin, index) => {
      const priceChangeClass = coin.price_change_percentage_24h >= 0 ? "green" : "red";
      const safeId = `sparkline-${coin.id.replace(/[^a-zA-Z0-9]/g, '')}`;

      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${(page - 1) * 50 + index + 1}</td>
        <td><img src="${coin.image}" class="crypto-img" alt="${coin.name}"> ${coin.name}</td>
        <td>
          $${coin.current_price.toLocaleString()}<br>
          <canvas id="${safeId}" width="80" height="30"></canvas>
        </td>
        <td class="${priceChangeClass}">${coin.price_change_percentage_24h.toFixed(2)}%</td>
      `;
      tbody.appendChild(row);

      if (coin.sparkline_in_7d && Array.isArray(coin.sparkline_in_7d.price)) {
        const ctx = document.getElementById(safeId).getContext('2d');
        new Chart(ctx, {
          type: 'line',
          data: {
            labels: coin.sparkline_in_7d.price.map((_, i) => i),
            datasets: [{
              data: coin.sparkline_in_7d.price,
              borderColor: priceChangeClass === "green" ? "#0f9d58" : "#db4437",
              borderWidth: 1,
              fill: false,
              pointRadius: 0
            }]
          },
          options: {
            responsive: false,
            maintainAspectRatio: false,
            elements: { line: { tension: 0.2 } },
            plugins: { legend: { display: false } },
            scales: { x: { display: false }, y: { display: false } }
          }
        });
      }
    });

    pageNumber.textContent = page;
    prevBtn.disabled = page <= 1;
    nextBtn.disabled = page >= maxPage;

  } catch (err) {
    console.error("Error fetching data:", err);
  }
}

prevBtn.onclick = () => {
  if (currentPage > 1) {
    currentPage--;
    fetchData(currentPage);
  }
};

nextBtn.onclick = () => {
  if (currentPage < maxPage) {
    currentPage++;
    fetchData(currentPage);
  }
};

fetchData(currentPage);
setInterval(() => fetchData(currentPage), 120000);

predictBtn.addEventListener("click", async () => {
  const selectedModel = modelSelect.value; // get selected model from dropdown
  let endpoint = "";

  if(selectedModel === "lstm"){
    endpoint = "http://localhost:5000/predict_lstm";
  } else if(selectedModel === "xgboost") {
    endpoint = "http://localhost:5000/predict_xgb";
  } else {
    result.textContent = "❌ Invalid Model Selection";
    result.style.color = "red";
    return;
  }

  result.textContent = "⏳ Predicting...";
  result.style.color = "gray";

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    });

    const data = await response.json();

    if (data.predicted_price) {
      result.textContent = `💰 Predicted Close Price: $${parseFloat(data.predicted_price).toFixed(2)}`;
      result.style.color = "green";
    } else if(data.error){
      result.textContent = `❌ Error: ${data.error}`;
      result.style.color = "red";
    } else {
      result.textContent = "❌ Unexpected error occurred.";
      result.style.color = "red";
    }

  } catch (error) {
    console.error("Error:", error);
    result.textContent = "❌ Could not connect to backend.";
    result.style.color = "red";
  }
});

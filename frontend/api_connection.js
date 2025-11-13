const tbody = document.getElementById("crypto-body");
const pageNumber = document.getElementById("page-number");
const prevBtn = document.getElementById("prev-btn");
const nextBtn = document.getElementById("next-btn");

const modelSelect = document.getElementById("model-select");
const predictBtn = document.getElementById("predict-btn");
const result = document.getElementById("prediction-result");

let currentPage = 1;
const maxPage = 50;

let chartInstance = null;
const chartCtx = document.getElementById("prediction-chart").getContext("2d");

// Fetch coins
async function fetchData(page = 1) {
    try {
        const url = 
          `https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=50&page=${page}&sparkline=true`;

        const response = await fetch(url);
        const data = await response.json();

        tbody.innerHTML = "";

        data.forEach((coin, index) => {
            const changePos = coin.price_change_percentage_24h >= 0;
            const safeId = `spark-${coin.id.replace(/[^a-zA-Z0-9]/g, '')}`;

            const row = document.createElement("tr");
            row.innerHTML = `
                <td>${(page - 1) * 50 + index + 1}</td>
                <td><img src="${coin.image}" class="crypto-img"> ${coin.name}</td>
                <td>$${coin.current_price.toLocaleString()}</td>
                <td><canvas id="${safeId}" width="80" height="30"></canvas></td>
                <td class="${changePos ? 'green' : 'red'}">
                    ${coin.price_change_percentage_24h?.toFixed(2)}%
                </td>
            `;

            tbody.appendChild(row);

            // Sparkline
            if (coin.sparkline_in_7d) {
                const ctx = document.getElementById(safeId).getContext("2d");

                new Chart(ctx, {
                    type: "line",
                    data: {
                        labels: coin.sparkline_in_7d.price.map((_, i) => i),
                        datasets: [{
                            data: coin.sparkline_in_7d.price,
                            borderColor: changePos ? "#0f9d58" : "#db4437",
                            borderWidth: 1,
                            pointRadius: 0,
                            fill: false,
                            tension: 0.3
                        }]
                    },
                    options: {
                        responsive: false,
                        scales: { x: { display: false }, y: { display: false } },
                        plugins: { legend: { display: false } }
                    }
                });
            }
        });

        pageNumber.textContent = currentPage;
    } catch (err) {
        console.error(err);
    }
}

// Pagination
prevBtn.onclick = () => { if (currentPage > 1) fetchData(--currentPage); };
nextBtn.onclick = () => { if (currentPage < maxPage) fetchData(++currentPage); };

fetchData(currentPage);


// ============ Prediction + Graph ============
predictBtn.addEventListener("click", async () => {
    const model = modelSelect.value;

    result.textContent = "⏳ Predicting...";
    result.style.color = "gray";

    try {
        const res = await fetch("http://localhost:5000/predict_with_history", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model })
        });

        const data = await res.json();

        if (data.error) {
            result.textContent = "❌ " + data.error;
            result.style.color = "red";
            return;
        }

        // LAST 30 DAYS ONLY
        let history = data.history.slice(-30);
        const predicted = data.predicted_price;

        result.textContent = `💰 Predicted Next-Day Price: $${predicted.toFixed(2)}`;
        result.style.color = "#1b7b1b";

        // SHOW THE CHART
        document.getElementById("chart-section").style.display = "block";

        const labels = history.map((_, i) => i);
        const fullPred = [...history, predicted];

        if (chartInstance) chartInstance.destroy();

        chartInstance = new Chart(chartCtx, {
            type: "line",
            data: {
                labels: [...labels, labels.length],
                datasets: [
                    {
                        label: "Last 30 Days",
                        data: history,
                        borderColor: "#007bff",
                        borderWidth: 2,
                        pointRadius: 2,
                        tension: 0.3,
                        fill: false
                    },
                    {
                        label: "Predicted Price",
                        data: fullPred,
                        borderColor: "#ff0033",
                        borderWidth: 2,
                        tension: 0.3,
                        fill: false,
                        pointRadius: ctx =>
                          ctx.p0DataIndex === fullPred.length - 1 ? 10 : 0,
                        pointBackgroundColor: "#ff0033"
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    x: { display: false },
                    y: { ticks: { callback: val => val.toLocaleString() } }
                }
            }
        });

    } catch (err) {
        console.error(err);
        result.textContent = "❌ Backend not responding";
        result.style.color = "red";
    }
});

// Generate synthetic data
function generateData(type, n=100) {
    const X = [];
    const y = [];
    if (type === 'classification') {
        for (let i = 0; i < n / 2; i++) {
            X.push([Math.random() * 4 + 1, Math.random() * 4 + 1]);
            y.push(0);
        }
        for (let i = 0; i < n / 2; i++) {
            X.push([Math.random() * 4 + 5, Math.random() * 4 + 5]);
            y.push(1);
        }
    } else if (type === 'clustering') {
        for (let i = 0; i < n; i++) {
            X.push([Math.random() * 10, Math.random() * 10]);
        }
    }
    return { X, y };
}

const svmData = generateData('classification');
const kmeansData = generateData('clustering');
const annData = generateData('classification');

// UI Update Listeners
document.getElementById('svm-c').addEventListener('input', (e) => {
    document.getElementById('svm-c-val').innerText = e.target.value;
});
document.getElementById('kmeans-k').addEventListener('input', (e) => {
    document.getElementById('kmeans-k-val').innerText = e.target.value;
});
document.getElementById('ann-hidden').addEventListener('input', (e) => {
    document.getElementById('ann-hidden-val').innerText = e.target.value;
});
document.getElementById('ann-epochs').addEventListener('input', (e) => {
    document.getElementById('ann-epochs-val').innerText = e.target.value;
});

// API Calls
async function trainSVM() {
    const kernel = document.getElementById('svm-kernel').value;
    const C = parseFloat(document.getElementById('svm-c').value);
    const cv = document.getElementById('svm-cv').checked;

    document.getElementById('train-svm-btn').classList.add('loading');

    try {
        const response = await fetch('/api/train/svm', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ X: svmData.X, y: svmData.y, kernel, C, cv })
        });
        const result = await response.json();

        if (result.metrics) {
            document.getElementById('stat-accuracy').innerText = (result.metrics.accuracy * 100).toFixed(1) + '%';
            document.getElementById('stat-precision').innerText = (result.metrics.precision * 100).toFixed(1) + '%';
            document.getElementById('stat-recall').innerText = (result.metrics.recall * 100).toFixed(1) + '%';
        }

        drawSVMPlot(svmData.X, svmData.y, result);
    } catch (e) {
        console.error(e);
    } finally {
        document.getElementById('train-svm-btn').classList.remove('loading');
    }
}

async function trainKMeans() {
    const k = parseInt(document.getElementById('kmeans-k').value);

    document.getElementById('train-kmeans-btn').classList.add('loading');

    try {
        const response = await fetch('/api/train/kmeans', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ X: kmeansData.X, n_clusters: k })
        });
        const result = await response.json();
        drawKMeansPlot(kmeansData.X, result);
    } catch (e) {
        console.error(e);
    } finally {
        document.getElementById('train-kmeans-btn').classList.remove('loading');
    }
}

async function trainANN() {
    const hidden_size = parseInt(document.getElementById('ann-hidden').value);
    const epochs = parseInt(document.getElementById('ann-epochs').value);

    document.getElementById('train-ann-btn').classList.add('loading');

    try {
        const response = await fetch('/api/train/ann', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ X: annData.X, y: annData.y, hidden_size, epochs })
        });
        const result = await response.json();
        drawANNChart(result.loss_history);
    } catch (e) {
        console.error(e);
    } finally {
        document.getElementById('train-ann-btn').classList.remove('loading');
    }
}

document.getElementById('train-svm-btn').addEventListener('click', trainSVM);
document.getElementById('train-kmeans-btn').addEventListener('click', trainKMeans);
document.getElementById('train-ann-btn').addEventListener('click', trainANN);

// D3 drawing functions
function drawSVMPlot(X, y, result) {
    const container = document.getElementById('svm-plot');
    container.innerHTML = '';

    const width = container.clientWidth;
    const height = container.clientHeight;

    const svg = d3.select('#svm-plot')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    const margin = {top: 20, right: 20, bottom: 30, left: 40};
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const xMin = d3.min(X, d => d[0]) - 1;
    const xMax = d3.max(X, d => d[0]) + 1;
    const yMin = d3.min(X, d => d[1]) - 1;
    const yMax = d3.max(X, d => d[1]) + 1;

    const xScale = d3.scaleLinear().domain([xMin, xMax]).range([0, w]);
    const yScale = d3.scaleLinear().domain([yMin, yMax]).range([h, 0]);

    // Draw decision boundary contour if data available
    if (result && result.xx && result.yy && result.Z) {
        const contourData = [];
        for(let i=0; i<result.Z.length; i++) {
            for(let j=0; j<result.Z[i].length; j++) {
                contourData.push({
                    x: result.xx[i][j],
                    y: result.yy[i][j],
                    z: result.Z[i][j]
                });
            }
        }

        // Very basic visual representation using circles since contour generator is complex
        g.selectAll('.decision-bg')
            .data(contourData)
            .enter().append('rect')
            .attr('class', 'decision-bg')
            .attr('x', d => xScale(d.x))
            .attr('y', d => yScale(d.y))
            .attr('width', 4) // adjust based on grid size
            .attr('height', 4)
            .attr('fill', d => d.z === 1 ? 'rgba(255, 0, 153, 0.1)' : 'rgba(56, 189, 248, 0.1)'); // Synthwave colors
    }

    // Support Vectors Highlight
    if (result && result.support_vectors) {
        g.selectAll('.support-vector')
            .data(result.support_vectors)
            .enter().append('circle')
            .attr('class', 'support-vector')
            .attr('cx', d => xScale(d[0]))
            .attr('cy', d => yScale(d[1]))
            .attr('r', 8)
            .attr('fill', 'none')
            .attr('stroke', '#e0e0e0')
            .attr('stroke-width', 2);
    }

    // Data points
    g.selectAll('.dot')
        .data(X)
        .enter().append('circle')
        .attr('class', 'dot')
        .attr('cx', d => xScale(d[0]))
        .attr('cy', d => yScale(d[1]))
        .attr('r', 5)
        .attr('fill', (d, i) => y[i] === 1 ? '#ff0099' : '#38bdf8') // Synthwave pink/blue
        .attr('stroke', '#fff')
        .attr('stroke-width', 1);

    // Axes
    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale));
    g.append('g').call(d3.axisLeft(yScale));
}

function drawKMeansPlot(X, result) {
    const container = document.getElementById('kmeans-plot');
    container.innerHTML = '';

    const width = container.clientWidth;
    const height = container.clientHeight;

    const svg = d3.select('#kmeans-plot')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    const margin = {top: 20, right: 20, bottom: 30, left: 40};
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const xMin = d3.min(X, d => d[0]) - 1;
    const xMax = d3.max(X, d => d[0]) + 1;
    const yMin = d3.min(X, d => d[1]) - 1;
    const yMax = d3.max(X, d => d[1]) + 1;

    const xScale = d3.scaleLinear().domain([xMin, xMax]).range([0, w]);
    const yScale = d3.scaleLinear().domain([yMin, yMax]).range([h, 0]);

    const colors = ['#ff0099', '#38bdf8', '#ffb86c', '#50fa7b', '#bd93f9', '#ff5555', '#f1fa8c', '#8be9fd', '#6272a4', '#ff79c6'];

    // Data points
    g.selectAll('.dot')
        .data(X)
        .enter().append('circle')
        .attr('class', 'dot')
        .attr('cx', d => xScale(d[0]))
        .attr('cy', d => yScale(d[1]))
        .attr('r', 5)
        .attr('fill', (d, i) => result && result.labels ? colors[result.labels[i] % colors.length] : '#888')
        .attr('stroke', '#fff')
        .attr('stroke-width', 1);

    // Centroids
    if (result && result.cluster_centers) {
        g.selectAll('.centroid')
            .data(result.cluster_centers)
            .enter().append('path')
            .attr('class', 'centroid')
            .attr('d', d3.symbol().type(d3.symbolCross).size(150))
            .attr('transform', d => `translate(${xScale(d[0])},${yScale(d[1])})`)
            .attr('fill', '#fff')
            .attr('stroke', '#000')
            .attr('stroke-width', 1);
    }

    // Axes
    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale));
    g.append('g').call(d3.axisLeft(yScale));
}

// Initial draw
drawSVMPlot(svmData.X, svmData.y, null);
drawKMeansPlot(kmeansData.X, null);

let annChart = null;
function drawANNChart(loss_history) {
    const ctx = document.getElementById('ann-loss-chart').getContext('2d');

    if (annChart) {
        annChart.destroy();
    }

    const labels = loss_history.map((_, i) => `Epoch ${i + 1}`);

    annChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Training Loss',
                data: loss_history,
                borderColor: 'rgba(56, 189, 248, 1)', // Tailwind text-accent-like color
                backgroundColor: 'rgba(56, 189, 248, 0.2)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Loss',
                        color: 'rgba(255, 255, 255, 0.7)'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Epochs',
                        color: 'rgba(255, 255, 255, 0.7)'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        maxTicksLimit: 10
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                }
            }
        }
    });
}

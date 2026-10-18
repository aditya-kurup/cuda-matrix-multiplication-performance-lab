(() => {
  const $ = (id) => document.getElementById(id);
  const A = Array.from({ length: 16 }, (_, i) => i + 1);
  const B = Array.from({ length: 16 }, (_, i) => (i % 4) + 1 + Math.floor(i / 4));
  const at = (matrix, row, col) => matrix[row * 4 + col];
  const state = { mode: 'naive', step: 0, row: 1, col: 2 };

  function matrix(container, values, type) {
    container.replaceChildren();
    values.forEach((value, index) => {
      const row = Math.floor(index / 4), col = index % 4;
      const cell = document.createElement(type === 'c' ? 'button' : 'span');
      cell.className = 'cell';
      cell.textContent = type === 'c' ? '·' : value;
      cell.title = `${type.toUpperCase()}[${row},${col}]`;
      cell.dataset.row = row;
      cell.dataset.col = col;
      if (type === 'c') {
        cell.type = 'button';
        cell.setAttribute('aria-label', `Select output C row ${row + 1}, column ${col + 1}`);
        cell.addEventListener('click', () => { state.row = row; state.col = col; state.step = 0; render(); });
      }
      container.appendChild(cell);
    });
  }

  function mark(container, predicate, className) {
    container.querySelectorAll('.cell').forEach((cell) => {
      if (predicate(Number(cell.dataset.row), Number(cell.dataset.col))) cell.classList.add(className);
    });
  }

  function miniGrid(container, values, blue) {
    container.className = `mini-grid${blue ? ' blue' : ''}`;
    container.replaceChildren(...values.map((value) => {
      const cell = document.createElement('span');
      cell.textContent = value;
      return cell;
    }));
  }

  function render() {
    for (const container of [$('matrix-a'), $('matrix-b'), $('matrix-c')]) {
      container.querySelectorAll('.cell').forEach((cell) => cell.classList.remove('read-a', 'read-b', 'tile-a', 'tile-b', 'selected', 'output-block'));
    }
    const tiled = state.mode === 'tiled';
    $('mode-naive').classList.toggle('active', !tiled);
    $('mode-tiled').classList.toggle('active', tiled);
    $('mode-naive').setAttribute('aria-pressed', String(!tiled));
    $('mode-tiled').setAttribute('aria-pressed', String(tiled));
    $('step-count').textContent = `${state.step + 1} / 4`;
    $('previous-step').disabled = state.step === 0;
    $('next-step').disabled = state.step === 3;
    mark($('matrix-c'), (r, c) => r === state.row && c === state.col, 'selected');
    if (!tiled) {
      const k = state.step;
      mark($('matrix-a'), (r, c) => r === state.row && c === k, 'read-a');
      mark($('matrix-b'), (r, c) => r === k && c === state.col, 'read-b');
      $('visual-phase').textContent = `NAIVE / INNER INDEX ${k}`;
      $('visual-explanation').textContent = `The thread for C[${state.row},${state.col}] reads A[${state.row},${k}] and B[${k},${state.col}], then adds their product to its accumulator.`;
      $('shared-panel').hidden = true;
    } else {
      const base = Math.floor(state.step / 2) * 2;
      const loading = state.step % 2 === 0;
      const rowBase = Math.floor(state.row / 2) * 2;
      const colBase = Math.floor(state.col / 2) * 2;
      mark($('matrix-a'), (r, c) => r >= rowBase && r < rowBase + 2 && c >= base && c < base + 2, 'tile-a');
      mark($('matrix-b'), (r, c) => r >= base && r < base + 2 && c >= colBase && c < colBase + 2, 'tile-b');
      mark($('matrix-c'), (r, c) => r >= rowBase && r < rowBase + 2 && c >= colBase && c < colBase + 2, 'output-block');
      miniGrid($('shared-a'), [at(A,rowBase,base),at(A,rowBase,base+1),at(A,rowBase+1,base),at(A,rowBase+1,base+1)], false);
      miniGrid($('shared-b'), [at(B,base,colBase),at(B,base,colBase+1),at(B,base+1,colBase),at(B,base+1,colBase+1)], true);
      $('visual-phase').textContent = `TILED / K SEGMENT ${base}–${base + 1} / ${loading ? 'LOAD' : 'REUSE'}`;
      $('visual-explanation').textContent = loading
        ? 'Threads in this output block cooperatively load the highlighted A and B tiles. A barrier makes the shared values ready for every thread.'
        : `The same loaded values contribute to four outputs in the highlighted 2 × 2 block. Threads synchronize again before the next tile overwrites shared memory.`;
      $('shared-panel').hidden = false;
    }
  }

  matrix($('matrix-a'), A, 'a');
  matrix($('matrix-b'), B, 'b');
  matrix($('matrix-c'), Array(16).fill(0), 'c');
  $('mode-naive').addEventListener('click', () => { state.mode = 'naive'; state.step = 0; render(); });
  $('mode-tiled').addEventListener('click', () => { state.mode = 'tiled'; state.step = 0; render(); });
  $('previous-step').addEventListener('click', () => { state.step = Math.max(0, state.step - 1); render(); });
  $('next-step').addEventListener('click', () => { state.step = Math.min(3, state.step + 1); render(); });
  render();

  function sourceLinks() {
    const pieces = location.pathname.split('/').filter(Boolean);
    const project = location.hostname.endsWith('.github.io') ? pieces[0] : null;
    const owner = project ? location.hostname.split('.')[0] : null;
    const root = owner && project ? `https://github.com/${owner}/${project}/blob/main/` : '../';
    document.querySelectorAll('[data-source]').forEach((link) => { link.href = root + link.dataset.source; });
  }
  sourceLinks();

  const format = (number, digits = 3) => Number(number).toFixed(digits);
  function valid(data) {
    return data && data.schema_version === 1 && data.environment?.gpu_name &&
      Number.isInteger(data.methodology?.trials) && Array.isArray(data.results) &&
      data.results.length > 0 && data.results.every((shape) =>
        Number.isInteger(shape.m) && Number.isInteger(shape.k) && Number.isInteger(shape.n) &&
        Number.isFinite(shape.h2d?.median_ms) && Array.isArray(shape.kernels) &&
        ['naive','tiled'].every((name) => shape.kernels.some((kernel) =>
          kernel.name === name && Number.isFinite(kernel.timing?.median_ms) &&
          kernel.timing.median_ms > 0 && Number.isFinite(kernel.gflops) &&
          typeof kernel.correctness?.passed === 'boolean')));
  }

  function showShape(data, index) {
    const shape = data.results[index];
    const naive = shape.kernels.find((kernel) => kernel.name === 'naive');
    const tiled = shape.kernels.find((kernel) => kernel.name === 'tiled');
    const maxTime = Math.max(naive.timing.median_ms, tiled.timing.median_ms);
    $('gpu-name').textContent = data.environment.gpu_name;
    $('trial-count').textContent = String(data.methodology.trials);
    $('correctness-status').textContent = naive.correctness.passed && tiled.correctness.passed ? 'Both passed' : 'Failure recorded';
    $('transfer-time').textContent = `${format(shape.h2d.median_ms)} ms`;
    $('run-date').textContent = data.generated_at_utc ? `Run: ${data.generated_at_utc.replace('T',' ').replace('Z',' UTC')}` : '';
    const ratio = naive.timing.median_ms / tiled.timing.median_ms;
    $('speedup-value').textContent = `${format(ratio, 2)}×`;
    $('comparison-text').textContent = ratio >= 1 ? 'Tiled kernel relative to naive, measured on this shape.' : 'Tiled kernel is slower on this shape.';
    const chart = $('timing-chart');
    chart.replaceChildren();
    const heading = document.createElement('p');
    heading.className = 'chart-title';
    heading.textContent = 'Median kernel time';
    chart.appendChild(heading);
    for (const kernel of [naive, tiled]) {
      const row = document.createElement('div');
      row.className = 'bar-row';
      const label = document.createElement('span');
      label.className = 'bar-name';
      label.textContent = kernel.name === 'naive' ? 'Naive' : 'Tiled';
      const track = document.createElement('div');
      track.className = 'bar-track';
      const fill = document.createElement('div');
      fill.className = `bar-fill ${kernel.name}`;
      fill.style.width = `${100 * kernel.timing.median_ms / maxTime}%`;
      track.appendChild(fill);
      const value = document.createElement('span');
      value.className = 'bar-value';
      value.textContent = `${format(kernel.timing.median_ms)} ms`;
      row.append(label, track, value);
      chart.appendChild(row);
    }
    const axis = document.createElement('p');
    axis.className = 'chart-axis';
    axis.textContent = 'Shorter bar = lower median execution time';
    chart.appendChild(axis);
    const details = $('details-table');
    details.replaceChildren();
    for (const kernel of [naive, tiled]) {
      const item = document.createElement('div');
      item.className = 'detail-row';
      const title = document.createElement('strong');
      title.textContent = `${kernel.name === 'naive' ? 'Naive' : 'Tiled'} · ${format(kernel.gflops, 2)} GFLOP/s`;
      const range = document.createElement('span');
      range.textContent = `Trial range ${format(kernel.timing.min_ms)}–${format(kernel.timing.max_ms)} ms · max abs error ${Number(kernel.correctness.max_absolute_error).toExponential(2)}`;
      item.append(title, range);
      details.appendChild(item);
    }
  }

  fetch('results/benchmark.json', { cache: 'no-store' })
    .then((response) => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); })
    .then((data) => {
      if (!valid(data)) throw new Error('Invalid benchmark schema');
      const select = $('shape-select');
      data.results.forEach((shape, index) => {
        const option = document.createElement('option');
        option.value = String(index);
        option.textContent = `${shape.m} × ${shape.k} · ${shape.k} × ${shape.n}`;
        select.appendChild(option);
      });
      select.addEventListener('change', () => showShape(data, Number(select.value)));
      $('pending').hidden = true;
      $('result-content').hidden = false;
      showShape(data, 0);
    })
    .catch((error) => { console.info('Benchmarks pending a GPU run:', error.message); });
})();

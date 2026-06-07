#!/bin/bash
# 批量复刻 KnowCS 其余 5 个模块页面为 .pen 设计文件
cd "$(dirname "$0")/.."

SHARED="Replicate this existing web app UI as a design, matching the attached reference screenshot faithfully — same layout, colors, typography and content. Shared shell: top header 'CS/AI Interactive Learning Lab' with COMP2211 subtitle, V2.0 badge, environment status pill and 中文 language toggle; left sidebar 'Knowledge Core' with 6 module nav entries (NumPy Mechanism, Backpropagation, Kernel Lab, Bayes Basics, Naive Bayes, K-Nearest Neighbors) plus blue gradient 'Exam Tip' card at bottom; footer bar with copyright and Latency/Render status."

run_one() {
  local name="$1"; local prompt="$2"
  for attempt in 1 2; do
    echo "=== [$name] attempt $attempt $(date) ==="
    if pencil --out "designs/knowcs-$name.pen" \
        --prompt "$SHARED $prompt" \
        --prompt-file "designs/ref-$name.png" \
        --export "designs/knowcs-$name.png" --export-scale 2; then
      echo "=== [$name] OK ==="
      return 0
    fi
    echo "=== [$name] FAILED attempt $attempt ==="
    sleep 10
  done
  echo "=== [$name] GAVE UP ==="
  return 1
}

run_one backprop "Active nav: Backpropagation. Main content: 'Backpropagation & Gradient Update' module — section 'Chain Rule Tracker' with a dark navy card showing a neural network diagram (input node, weight w, hidden node, red Error node, connecting edges), below it an 'Error Value [Target - Out]' slider control and a red 'Track Gradient' button; on the right a 'Derivation Notebook (COMP2211 Core)' white card with 3 steps, each containing KaTeX-style partial-derivative chain rule formulas (∂E/∂w expansion, sigmoid derivative, weight update rule), and a yellow Senior's Advice callout at the bottom about backpropagation being 'finding the culprit'."

run_one kernel "Active nav: Kernel Lab. Main content: 'Kernel Laboratory' module — subtitle 'Real-time Simulation of Edge Feature Extraction'; a toolbar card with 4 kernel preset buttons (Sobel-X active in blue, Sobel-Y, Laplacian, Identity) and an OPERATION formula label on the right; below, a three-panel diagram: 'INPUT IMAGE I' black pixel grid with a gray T-shape, × symbol, 'KERNEL K' 3x3 editable grid with values (-1 0 1 / -2 0 2 / -1 0 1) on a light indigo card, = symbol, 'OUTPUT FEATURE MAP' black/white/gray pixel grid showing detected vertical edges; and a yellow Senior's Advice callout about convolution kernels being local feature filters."

run_one bayes-basics "Active nav: Bayes Basics. Main content: 'Bayes Theorem Basics' module — section '1. Bayes Rule Core Analysis' with the large Bayes formula P(H|E) = P(H) × P(E|H)/P(E) in KaTeX style, below it a 2x2 grid of four colored definition cards: Posterior (blue), Prior (purple), Likelihood (green), Marginal (orange), each with a short description; section '2. Practice: Smoke & Fire Case (Notes P.15)' with three probability sliders on the left (Prob P(Fire) red, Prob P(Smoke) blue, Prob P(Smoke|Fire) purple, each with value badge), and on the right a dark navy result card 'Prob of fire given smoke' showing large red '9.00%' with the calculation formula below; and a yellow Senior's Advice callout about classic exam cases (Evidence ↑ Conclusion, Prior matters)."

run_one naive-bayes "Active nav: Naive Bayes. Main content: 'Naive Bayes Inference' module — subtitle 'Classifier with Feature Conditional Independence'; left card 'FEATURE INPUT' with 4 labeled dropdowns (BP: High, Fever: No, Diabetes: Yes, Vomit: Yes), a 'SMOOTHING λ:' number input showing 0, and a 'PRODUCT MODE' button; top right a dark navy card 'CLASSIFICATION RESULT (NOTES P.29)' with two probability bars: CLASS: YES 20.46% (blue bar) and CLASS: NO 79.54% (gray bar); below it a white card 'LIKELIHOOD CHAIN' with two columns 'PROBABILITIES FOR YES' and 'PROBABILITIES FOR NO', each listing 5 KaTeX-style conditional probabilities P(Z=Yes) 0.643, P(BP|yes) 0.222 etc.; and a yellow Senior's Advice callout about the Zero Frequency Problem."

run_one knn "Active nav: K-Nearest Neighbors. Main content: 'K-Nearest Neighbors' module — subtitle 'Distance Metric & Decision Boundary'; left: a large white scatter-plot card with red and blue data points, a highlighted 'Your Sample' test point with circular neighbor radius, and dashed crosshair lines; top right: a dark navy card 'Model Complexity vs Error Rate' with a blue dotted U-shaped curve chart and Overfitting/Underfitting labels at bottom; bottom right: a white card 'K Classification Voting Panel' with 'Parameter K:' control and two vote count boxes (blue 4 vs gray 1) with a slider; and a yellow Senior's Advice callout about small K overfitting / large K underfitting and Standardized distance."

echo "ALL DONE $(date)"

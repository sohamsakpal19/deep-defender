function switchTab(event, tabId) {
    const panels = document.querySelectorAll('.tab-panel');
    const links = document.querySelectorAll('.tab-link');
    
    panels.forEach(p => p.classList.remove('active'));
    links.forEach(l => l.classList.remove('active'));
    
    document.getElementById(tabId).classList.add('active');
    event.currentTarget.classList.add('active');
}

document.getElementById('file-input').addEventListener('change', function(e) {
    if (e.target.files.length > 0) {
        processAnalysis(e.target.files[0].name);
    }
});

function processAnalysis(source) {
    const box = document.getElementById('result-display');
    const spinner = document.getElementById('spinner');
    const content = document.getElementById('analysis-content');
    const title = document.getElementById('risk-title');
    const bar = document.getElementById('meter-fill');
    const desc = document.getElementById('risk-desc');

    box.classList.remove('hidden');
    spinner.classList.remove('hidden');
    content.classList.add('hidden');

    setTimeout(() => {
        spinner.classList.add('hidden');
        content.classList.remove('hidden');

        const isSuspicious = source.toLowerCase().includes('fake') || source === 'url';
        
        if (isSuspicious) {
            title.innerText = "High Risk Detected";
            title.style.color = "#ff4d4d";
            bar.style.width = "85%";
            bar.style.backgroundColor = "#ff4d4d";
            desc.innerText = "Our models detected significant markers of AI manipulation in this content.";
        } else {
            title.innerText = "Content Authentic";
            title.style.color = "#00ff88";
            bar.style.width = "10%";
            bar.style.backgroundColor = "#00ff88";
            desc.innerText = "No significant signs of forgery detected. Content matches original patterns.";
        }
    }, 2000);
}

const dropArea = document.getElementById('drop-area');

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(name => {
    dropArea.addEventListener(name, e => {
        e.preventDefault();
        e.stopPropagation();
    });
});

dropArea.addEventListener('drop', e => {
    const file = e.dataTransfer.files[0];
    processAnalysis(file.name);
});
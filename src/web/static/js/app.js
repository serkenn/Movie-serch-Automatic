/* Global utility functions */

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

async function fetchJSON(url) {
    const resp = await fetch(url);
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: resp.statusText }));
        throw new Error(err.error || resp.statusText);
    }
    return resp.json();
}

async function refreshNetworkBadge() {
    const badge = document.getElementById('network-badge');
    if (!badge) return;
    try {
        const data = await fetchJSON('/api/network/status');
        if (data.error) {
            badge.className = 'network-badge error';
            badge.textContent = `Network: ${data.error}`;
            return;
        }
        const loc = [data.city, data.region, data.country].filter(Boolean).join(', ') || '-';
        badge.className = data.warning ? 'network-badge warning' : 'network-badge';
        badge.textContent = `IP ${data.effective_ip || '-'} | ${loc}`;
    } catch (e) {
        badge.className = 'network-badge error';
        badge.textContent = `Network: ${e.message}`;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    refreshNetworkBadge();
    setInterval(refreshNetworkBadge, 15000);
});

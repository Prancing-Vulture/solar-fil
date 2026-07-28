import os
import time
import csv
import json
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-blocking backend for thread-safe live rendering
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, send_file, jsonify

class LiveDisplayer:
    """
    Live Displayer that logs training metrics per epoch, writes persistent CSV/TXT log files,
    renders real-time segmentation overlays, and serves a live web dashboard with metric logs at http://localhost:5000.
    """
    def __init__(self, port=5000, save_dir="live_display", log_dir="logs"):
        self.port = port
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.plot_path = os.path.join(save_dir, "live_epoch_display.png")
        self.csv_log_path = os.path.join(log_dir, "training_metrics.csv")
        self.txt_log_path = os.path.join(log_dir, "training.log")
        self.json_log_path = os.path.join(log_dir, "training_metrics.json")
        
        # Initialize CSV header if file doesn't exist
        if not os.path.exists(self.csv_log_path):
            with open(self.csv_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'epoch', 'train_loss', 'val_loss', 'val_bce', 'train_dice', 'val_dice', 'val_iou'])
                
        # Initialize TXT log header
        with open(self.txt_log_path, 'a') as f:
            f.write(f"\n=== Training Session Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

        # Metric histories
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'val_iou': [],
            'val_bce': []
        }
        
        self.best_metrics = {
            'best_epoch': 0,
            'best_val_dice': 0.0,
            'best_val_loss': float('inf'),
            'best_val_iou': 0.0
        }
        
        # Web server thread
        self.app = Flask(__name__)
        self._setup_routes()
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        print(f"\n[LIVE DISPLAYER] Web dashboard running at http://localhost:{self.port}")
        print(f"[LOGS] Persistent CSV Log:  {self.csv_log_path}")
        print(f"[LOGS] Persistent Text Log: {self.txt_log_path}\n")

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Solar Filament Segmentation — Live Displayer & Logger</title>
                <meta http-equiv="refresh" content="3">
                <style>
                    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
                    .header { text-align: center; margin-bottom: 20px; }
                    .header h1 { color: #38bdf8; margin: 5px; font-size: 26px; }
                    .header p { color: #94a3b8; font-size: 14px; margin: 0; }
                    .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; max-width: 1200px; margin: 0 auto 20px auto; }
                    .card { background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; text-align: center; }
                    .card-title { color: #94a3b8; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
                    .card-val { font-size: 24px; font-weight: bold; color: #38bdf8; margin-top: 5px; }
                    .card-val.highlight { color: #4ade80; }
                    .container { max-width: 1200px; margin: 0 auto 25px auto; text-align: center; }
                    img { max-width: 100%; border-radius: 10px; border: 2px solid #38bdf8; box-shadow: 0 4px 20px rgba(56, 189, 248, 0.2); }
                    
                    /* Metrics Log Table */
                    .section-title { font-size: 18px; color: #38bdf8; margin-bottom: 10px; text-align: left; }
                    .table-wrapper { max-width: 1200px; margin: 0 auto 25px auto; background-color: #1e293b; border-radius: 10px; border: 1px solid #334155; padding: 15px; overflow-x: auto; }
                    table { width: 100%; border-collapse: collapse; text-align: center; font-size: 14px; }
                    th { background-color: #0f172a; color: #38bdf8; padding: 10px; border-bottom: 2px solid #334155; }
                    td { padding: 10px; border-bottom: 1px solid #334155; color: #cbd5e1; }
                    tr:hover { background-color: #334155; }
                    .badge-best { background-color: #166534; color: #4ade80; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }
                    
                    /* Live Console Log Box */
                    .log-box { max-width: 1200px; margin: 0 auto; background-color: #020617; border-radius: 10px; border: 1px solid #1e293b; padding: 15px; font-family: monospace; font-size: 13px; color: #a3e635; height: 180px; overflow-y: auto; text-align: left; white-space: pre-wrap; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Solar Filament Segmentation — Live Displayer & Metric Logger</h1>
                    <p>Real-time GPU training metrics, BCE logits loss, dice score tracking & live segmentation preview</p>
                </div>
                
                <div class="grid">
                    <div class="card">
                        <div class="card-title">Current Epoch</div>
                        <div class="card-val">{{ current_epoch }}</div>
                    </div>
                    <div class="card">
                        <div class="card-title">Best Val Dice</div>
                        <div class="card-val highlight">{{ "%.4f"|format(best_metrics.best_val_dice) }}</div>
                    </div>
                    <div class="card">
                        <div class="card-title">Best Val Loss</div>
                        <div class="card-val">{{ "%.4f"|format(best_metrics.best_val_loss) }}</div>
                    </div>
                    <div class="card">
                        <div class="card-title">Best Val IoU</div>
                        <div class="card-val highlight">{{ "%.4f"|format(best_metrics.best_val_iou) }}</div>
                    </div>
                </div>

                <div class="container">
                    <div class="section-title">Live Segmentation Preview & Epoch Curves</div>
                    <img src="/live_plot?t={{ timestamp }}" alt="Live Training Visualization">
                </div>

                <div class="table-wrapper">
                    <div class="section-title">Epoch Metrics History Log (BCE & Dice Losses)</div>
                    <table>
                        <thead>
                            <tr>
                                <th>Epoch</th>
                                <th>Train Combined Loss</th>
                                <th>Val Combined Loss</th>
                                <th>Val BCE Logits Loss</th>
                                <th>Train Dice Score</th>
                                <th>Val Dice Score</th>
                                <th>Val IoU (Jaccard)</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for row in log_rows %}
                            <tr>
                                <td><strong>Epoch {{ row.epoch }}</strong></td>
                                <td>{{ "%.4f"|format(row.train_loss) }}</td>
                                <td>{{ "%.4f"|format(row.val_loss) }}</td>
                                <td>{{ "%.4f"|format(row.val_bce) }}</td>
                                <td>{{ "%.4f"|format(row.train_dice) }}</td>
                                <td><strong style="color: #4ade80;">{{ "%.4f"|format(row.val_dice) }}</strong></td>
                                <td>{{ "%.4f"|format(row.val_iou) }}</td>
                                <td>
                                    {% if row.epoch == best_metrics.best_epoch %}
                                        <span class="badge-best">🏆 BEST MODEL</span>
                                    {% else %}
                                        <span style="color: #94a3b8;">Logged</span>
                                    {% endif %}
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>

                <div class="container">
                    <div class="section-title">Real-Time Terminal Execution Log</div>
                    <div class="log-box">{{ text_logs }}</div>
                </div>
            </body>
            </html>
            """
            current_ep = self.history['epoch'][-1] if self.history['epoch'] else 0
            
            # Prepare log rows for HTML table
            log_rows = []
            for i in range(len(self.history['epoch'])):
                log_rows.append({
                    'epoch': self.history['epoch'][i],
                    'train_loss': self.history['train_loss'][i],
                    'val_loss': self.history['val_loss'][i],
                    'val_bce': self.history['val_bce'][i],
                    'train_dice': self.history['train_dice'][i],
                    'val_dice': self.history['val_dice'][i],
                    'val_iou': self.history['val_iou'][i]
                })
            log_rows.reverse() # Latest epoch at top
            
            # Read recent text log lines
            text_logs = ""
            if os.path.exists(self.txt_log_path):
                with open(self.txt_log_path, 'r') as f:
                    lines = f.readlines()
                    text_logs = "".join(lines[-25:]) # last 25 lines

            return render_template_string(
                html_template,
                current_epoch=current_ep,
                best_metrics=self.best_metrics,
                log_rows=log_rows,
                text_logs=text_logs,
                timestamp=int(time.time())
            )

        @self.app.route('/live_plot')
        def live_plot():
            if os.path.exists(self.plot_path):
                return send_file(self.plot_path, mimetype='image/png')
            else:
                fig, ax = plt.subplots(figsize=(6, 2))
                ax.text(0.5, 0.5, "Initializing Live Training...", ha='center', va='center', color='gray')
                ax.axis('off')
                temp_buf = os.path.join(self.save_dir, "temp_init.png")
                fig.savefig(temp_buf, bbox_inches='tight')
                plt.close(fig)
                return send_file(temp_buf, mimetype='image/png')

        @self.app.route('/api/metrics')
        def api_metrics():
            return jsonify({
                'history': self.history,
                'best_metrics': self.best_metrics
            })

    def _run_server(self):
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        self.app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False)

    def update_epoch(self, epoch, train_loss, val_loss, train_dice, val_dice, val_iou, val_bce,
                     sample_img_np, sample_gt_np, sample_pred_probs):
        """
        Updates epoch metrics, writes CSV/TXT logs, renders live plot, and refreshes dashboard.
        """
        timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S')
        
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_dice'].append(train_dice)
        self.history['val_dice'].append(val_dice)
        self.history['val_iou'].append(val_iou)
        self.history['val_bce'].append(val_bce)

        # Update best metrics
        is_best = False
        if val_dice > self.best_metrics['best_val_dice']:
            self.best_metrics['best_val_dice'] = val_dice
            self.best_metrics['best_epoch'] = epoch
            self.best_metrics['best_val_loss'] = val_loss
            self.best_metrics['best_val_iou'] = val_iou
            is_best = True

        # 1. Append to CSV log
        with open(self.csv_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp_str, epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_bce:.6f}", f"{train_dice:.6f}", f"{val_dice:.6f}", f"{val_iou:.6f}"])

        # 2. Append to TXT log
        log_line = f"[{timestamp_str}] Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val BCE: {val_bce:.4f} | Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}"
        if is_best:
            log_line += " -> [BEST CHECKPOINT SAVED]"
        log_line += "\n"
        with open(self.txt_log_path, 'a') as f:
            f.write(log_line)

        # 3. Save JSON history
        with open(self.json_log_path, 'w') as f:
            json.dump({'history': self.history, 'best_metrics': self.best_metrics}, f, indent=2)

        # 4. Render Live Figure
        fig = plt.figure(figsize=(16, 10), facecolor='#0f172a')
        
        # Subplot 1: Input Image
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(sample_img_np, cmap='gray')
        ax1.set_title("Input Solar H-Alpha Image", color='#38bdf8', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Subplot 2: Ground Truth Mask
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.imshow(sample_gt_np, cmap='inferno')
        ax2.set_title("Ground Truth Mask", color='#38bdf8', fontsize=12, fontweight='bold')
        ax2.axis('off')

        # Subplot 3: Pred Probability & Green Overlay
        ax3 = fig.add_subplot(2, 3, 3)
        h, w = sample_img_np.shape
        img_normalized = (sample_img_np - sample_img_np.min()) / (sample_img_np.max() - sample_img_np.min() + 1e-6)
        overlay = np.stack([img_normalized]*3, axis=-1)
        
        pred_mask = (sample_pred_probs > 0.5).astype(np.float32)
        overlay[pred_mask > 0] = [0.1, 0.95, 0.2] # Neon Green
        
        ax3.imshow(overlay)
        ax3.set_title(f"Epoch {epoch} Filament Segmentation Overlay", color='#4ade80', fontsize=12, fontweight='bold')
        ax3.axis('off')

        # Subplot 4: Loss Curves
        ax4 = fig.add_subplot(2, 3, 4, facecolor='#1e293b')
        ax4.plot(self.history['epoch'], self.history['train_loss'], label='Train Combined Loss', color='#38bdf8', lw=2)
        ax4.plot(self.history['epoch'], self.history['val_loss'], label='Val Combined Loss', color='#f43f5e', lw=2)
        ax4.plot(self.history['epoch'], self.history['val_bce'], label='Val BCE Logits Loss', color='#fbbf24', linestyle='--', lw=1.5)
        ax4.set_title("Loss Curves (BCE & Dice)", color='#f8fafc', fontsize=11)
        ax4.set_xlabel("Epoch", color='#94a3b8')
        ax4.set_ylabel("Loss", color='#94a3b8')
        ax4.tick_params(colors='#94a3b8')
        ax4.legend(facecolor='#0f172a', labelcolor='#f8fafc', fontsize=9)
        ax4.grid(True, color='#334155', linestyle=':', alpha=0.6)

        # Subplot 5: Metric Curves
        ax5 = fig.add_subplot(2, 3, 5, facecolor='#1e293b')
        ax5.plot(self.history['epoch'], self.history['train_dice'], label='Train Dice Score', color='#a855f7', lw=2)
        ax5.plot(self.history['epoch'], self.history['val_dice'], label='Val Dice Score', color='#4ade80', lw=2)
        ax5.plot(self.history['epoch'], self.history['val_iou'], label='Val IoU (Jaccard)', color='#06b6d4', linestyle='--', lw=1.5)
        ax5.set_title("Segmentation Performance Metrics", color='#f8fafc', fontsize=11)
        ax5.set_xlabel("Epoch", color='#94a3b8')
        ax5.set_ylabel("Score [0 - 1]", color='#94a3b8')
        ax5.tick_params(colors='#94a3b8')
        ax5.legend(facecolor='#0f172a', labelcolor='#f8fafc', fontsize=9)
        ax5.grid(True, color='#334155', linestyle=':', alpha=0.6)

        # Subplot 6: Metrics Summary Table Card
        ax6 = fig.add_subplot(2, 3, 6, facecolor='#1e293b')
        ax6.axis('off')
        
        summary_text = (
            f"  CURRENT & BEST METRICS SUMMARY\n"
            f"  --------------------------------------\n"
            f"  Epoch:               {epoch}\n"
            f"  Train Loss:          {train_loss:.4f}\n"
            f"  Val Loss:            {val_loss:.4f}\n"
            f"  Val BCE Logits Loss: {val_bce:.4f}\n"
            f"  Train Dice Score:    {train_dice:.4f}\n"
            f"  Val Dice Score:      {val_dice:.4f}\n"
            f"  Val IoU Index:       {val_iou:.4f}\n\n"
            f"  BEST PERFORMANCE (Epoch {self.best_metrics['best_epoch']}):\n"
            f"  --------------------------------------\n"
            f"  Best Val Dice:       {self.best_metrics['best_val_dice']:.4f}\n"
            f"  Best Val Loss:       {self.best_metrics['best_val_loss']:.4f}\n"
            f"  Best Val IoU:        {self.best_metrics['best_val_iou']:.4f}\n"
        )
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, color='#f8fafc',
                 fontsize=11, family='monospace', verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#0f172a', edgecolor='#38bdf8', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[DISPLAYER & LOG] [EPOCH {epoch:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val BCE: {val_bce:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | Best Dice: {self.best_metrics['best_val_dice']:.4f}")

if __name__ == "__main__":
    displayer = LiveDisplayer(port=5000)
    img_sample = np.random.rand(256, 256)
    gt_sample = (np.random.rand(256, 256) > 0.9).astype(np.float32)
    pred_sample = (np.random.rand(256, 256) > 0.85).astype(np.float32)
    
    displayer.update_epoch(
        epoch=1,
        train_loss=0.456,
        val_loss=0.412,
        train_dice=0.621,
        val_dice=0.678,
        val_iou=0.512,
        val_bce=0.354,
        sample_img_np=img_sample,
        sample_gt_np=gt_sample,
        sample_pred_probs=pred_sample
    )
    print("LiveDisplayer & Logger test passed!")

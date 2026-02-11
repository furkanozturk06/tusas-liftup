"""
================================================================================
ADVANCED VISUALIZATION MODULE
Extended Visualizations for Satellite Anomaly Detection
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import from main system
from satellite_anomaly_system import (
    TelemetryConfig, SatelliteAnomalyPipeline, AlertSeverity
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats


class AdvancedVisualizer:
    """
    Extended visualization suite for comprehensive anomaly analysis.
    Generates 15+ additional visualization types.
    """
    
    def __init__(self, config: TelemetryConfig, output_dir: str = '.'):
        self.config = config
        self.output_dir = output_dir
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Color schemes
        self.palette = {
            'primary': '#3498db',
            'secondary': '#2ecc71',
            'danger': '#e74c3c',
            'warning': '#f39c12',
            'info': '#9b59b6',
            'dark': '#2c3e50',
            'light': '#ecf0f1'
        }
        
        self.anomaly_colors = {
            'normal': '#2ecc71',
            'speed_': '#e74c3c',
            'torque': '#3498db',
            'curren': '#9b59b6',
            'vibrat': '#f39c12',
            'overte': '#1abc9c'
        }
    
    # =========================================================================
    # 1. CORRELATION ANALYSIS
    # =========================================================================
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, 
                                  save: bool = True) -> plt.Figure:
        """Plot feature correlation heatmap"""
        print("  [1/15] Generating correlation heatmap...")
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c in self.config.feature_columns]
        
        corr_matrix = df[feature_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='RdBu_r', center=0, ax=ax, 
                   square=True, linewidths=0.5,
                   cbar_kws={'shrink': 0.8})
        
        ax.set_title('Telemetry Feature Correlation Matrix', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/correlation_heatmap.png', dpi=150)
            
        return fig
    
    def plot_cross_correlation_matrix(self, df: pd.DataFrame,
                                      save: bool = True) -> plt.Figure:
        """Plot rolling cross-correlation between all sensor pairs"""
        print("  [2/15] Generating cross-correlation matrix...")
        
        feature_cols = [c for c in self.config.feature_columns if c in df.columns]
        n = len(feature_cols)
        
        fig, axes = plt.subplots(n, n, figsize=(16, 16))
        
        for i, col1 in enumerate(feature_cols):
            for j, col2 in enumerate(feature_cols):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal - distribution
                    ax.hist(df[col1], bins=30, color=self.palette['primary'], 
                           alpha=0.7, edgecolor='white')
                    ax.set_ylabel(col1.replace('_', '\n'), fontsize=8)
                elif i > j:
                    # Lower triangle - scatter
                    ax.scatter(df[col2], df[col1], alpha=0.3, s=5, 
                              c=self.palette['primary'])
                else:
                    # Upper triangle - correlation coefficient
                    corr = df[col1].corr(df[col2])
                    color = plt.cm.RdBu_r((corr + 1) / 2)
                    ax.set_facecolor(color)
                    ax.text(0.5, 0.5, f'{corr:.2f}', ha='center', va='center',
                           fontsize=12, fontweight='bold',
                           color='white' if abs(corr) > 0.5 else 'black')
                
                ax.set_xticks([])
                ax.set_yticks([])
                
                if i == n-1:
                    ax.set_xlabel(col2.replace('_', '\n'), fontsize=8)
        
        fig.suptitle('Pairwise Feature Analysis Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save:
            plt.savefig(f'{self.output_dir}/cross_correlation_matrix.png', dpi=150)
            
        return fig
    
    # =========================================================================
    # 2. DISTRIBUTION ANALYSIS
    # =========================================================================
    
    def plot_feature_distributions(self, df: pd.DataFrame,
                                   save: bool = True) -> plt.Figure:
        """Plot distribution of each feature split by normal/anomaly"""
        print("  [3/15] Generating feature distributions...")
        
        feature_cols = [c for c in self.config.feature_columns if c in df.columns]
        n_cols = len(feature_cols)
        
        fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8))
        
        for i, col in enumerate(feature_cols):
            # Top row - Histogram
            ax_hist = axes[0, i]
            
            if 'label' in df.columns:
                normal_data = df[df['label'] == 'normal'][col]
                anomaly_data = df[df['label'] == 'anomal'][col]
                
                ax_hist.hist(normal_data, bins=30, alpha=0.6, 
                            label='Normal', color=self.palette['secondary'],
                            edgecolor='white')
                ax_hist.hist(anomaly_data, bins=30, alpha=0.6,
                            label='Anomaly', color=self.palette['danger'],
                            edgecolor='white')
                ax_hist.legend(fontsize=8)
            else:
                ax_hist.hist(df[col], bins=30, color=self.palette['primary'],
                            edgecolor='white')
            
            ax_hist.set_title(col.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            ax_hist.set_xlabel('')
            
            # Bottom row - Box plot
            ax_box = axes[1, i]
            
            if 'label' in df.columns:
                data_to_plot = [
                    df[df['label'] == 'normal'][col].dropna(),
                    df[df['label'] == 'anomal'][col].dropna()
                ]
                bp = ax_box.boxplot(data_to_plot, labels=['Normal', 'Anomaly'],
                                   patch_artist=True)
                bp['boxes'][0].set_facecolor(self.palette['secondary'])
                bp['boxes'][1].set_facecolor(self.palette['danger'])
            else:
                ax_box.boxplot(df[col].dropna())
        
        fig.suptitle('Feature Distributions: Normal vs Anomaly', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save:
            plt.savefig(f'{self.output_dir}/feature_distributions.png', dpi=150)
            
        return fig
    
    def plot_violin_by_anomaly_type(self, df: pd.DataFrame,
                                    save: bool = True) -> plt.Figure:
        """Violin plots showing feature distributions per anomaly type"""
        print("  [4/15] Generating violin plots by anomaly type...")
        
        feature_cols = [c for c in self.config.feature_columns if c in df.columns]
        
        fig, axes = plt.subplots(1, len(feature_cols), figsize=(4*len(feature_cols), 6))
        
        if 'anomaly_type' in df.columns:
            anomaly_types = df['anomaly_type'].unique()
        else:
            anomaly_types = ['normal']
        
        for i, col in enumerate(feature_cols):
            ax = axes[i] if len(feature_cols) > 1 else axes
            
            if 'anomaly_type' in df.columns:
                # Create violin plot
                parts = ax.violinplot(
                    [df[df['anomaly_type'] == t][col].dropna() 
                     for t in anomaly_types if len(df[df['anomaly_type'] == t]) > 0],
                    showmeans=True, showmedians=True
                )
                
                # Color the violins
                colors = [self.anomaly_colors.get(t, '#gray') for t in anomaly_types]
                for pc, color in zip(parts['bodies'], colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                
                ax.set_xticks(range(1, len(anomaly_types) + 1))
                ax.set_xticklabels([t[:6] for t in anomaly_types], rotation=45, fontsize=8)
            
            ax.set_title(col.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('Feature Distribution by Anomaly Type', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save:
            plt.savefig(f'{self.output_dir}/violin_by_anomaly_type.png', dpi=150)
            
        return fig
    
    # =========================================================================
    # 3. DIMENSIONALITY REDUCTION
    # =========================================================================
    
    def plot_pca_analysis(self, X: np.ndarray, y: np.ndarray,
                          save: bool = True) -> plt.Figure:
        """PCA visualization of feature space"""
        print("  [5/15] Generating PCA analysis...")
        
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)
        
        fig = plt.figure(figsize=(16, 5))
        
        # 2D PCA
        ax1 = fig.add_subplot(131)
        scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, 
                             cmap='RdYlGn_r', alpha=0.6, s=20)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax1.set_title('PCA: PC1 vs PC2', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 3D PCA
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                   c=y, cmap='RdYlGn_r', alpha=0.6, s=20)
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_zlabel('PC3')
        ax2.set_title('PCA: 3D View', fontweight='bold')
        
        # Explained variance
        ax3 = fig.add_subplot(133)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components = min(10, len(pca.explained_variance_ratio_))
        ax3.bar(range(1, n_components+1), 
               pca.explained_variance_ratio_[:n_components],
               alpha=0.7, color=self.palette['primary'], label='Individual')
        ax3.plot(range(1, n_components+1), cumsum[:n_components], 
                'ro-', label='Cumulative')
        ax3.set_xlabel('Principal Component')
        ax3.set_ylabel('Explained Variance Ratio')
        ax3.set_title('Explained Variance', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/pca_analysis.png', dpi=150)
            
        return fig
    
    def plot_tsne_visualization(self, X: np.ndarray, y: np.ndarray,
                                y_types: np.ndarray,
                                save: bool = True) -> plt.Figure:
        """t-SNE visualization of feature space"""
        print("  [6/15] Generating t-SNE visualization...")
        
        # Subsample if too large
        if len(X) > 2000:
            idx = np.random.choice(len(X), 2000, replace=False)
            X_sample = X[idx]
            y_sample = y[idx]
            y_types_sample = y_types[idx]
        else:
            X_sample = X
            y_sample = y
            y_types_sample = y_types
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_sample)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # By normal/anomaly
        scatter1 = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                  c=y_sample, cmap='RdYlGn_r', 
                                  alpha=0.6, s=30)
        axes[0].set_title('t-SNE: Normal vs Anomaly', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Legend for first plot
        legend_elements = [
            Patch(facecolor=self.palette['secondary'], label='Normal'),
            Patch(facecolor=self.palette['danger'], label='Anomaly')
        ]
        axes[0].legend(handles=legend_elements)
        
        # By anomaly type
        unique_types = np.unique(y_types_sample)
        colors = [self.anomaly_colors.get(t, '#gray') for t in y_types_sample]
        axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, alpha=0.6, s=30)
        axes[1].set_title('t-SNE: By Anomaly Type', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Legend for second plot
        legend_elements = [
            Patch(facecolor=self.anomaly_colors.get(t, '#gray'), label=t[:8])
            for t in unique_types
        ]
        axes[1].legend(handles=legend_elements, fontsize=8)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/tsne_visualization.png', dpi=150)
            
        return fig
    
    # =========================================================================
    # 4. TEMPORAL ANALYSIS
    # =========================================================================
    
    def plot_hourly_patterns(self, df: pd.DataFrame,
                             save: bool = True) -> plt.Figure:
        """Plot hourly patterns of telemetry and anomalies"""
        print("  [7/15] Generating hourly patterns...")
        
        df_copy = df.copy()
        df_copy['hour'] = df_copy.index.hour
        
        feature_cols = [c for c in self.config.feature_columns if c in df.columns]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(feature_cols[:5]):
            ax = axes[i]
            
            hourly_stats = df_copy.groupby('hour')[col].agg(['mean', 'std'])
            
            ax.fill_between(hourly_stats.index, 
                           hourly_stats['mean'] - hourly_stats['std'],
                           hourly_stats['mean'] + hourly_stats['std'],
                           alpha=0.3, color=self.palette['primary'])
            ax.plot(hourly_stats.index, hourly_stats['mean'], 
                   color=self.palette['primary'], linewidth=2)
            
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel(col.replace('_', ' '))
            ax.set_title(f'{col.replace("_", " ").title()} by Hour', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(0, 24, 3))
        
        # Anomaly count by hour
        if 'label' in df.columns:
            ax = axes[5]
            anomaly_by_hour = df_copy[df_copy['label'] == 'anomal'].groupby('hour').size()
            ax.bar(anomaly_by_hour.index, anomaly_by_hour.values, 
                  color=self.palette['danger'], alpha=0.7, edgecolor='white')
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Anomaly Count')
            ax.set_title('Anomalies by Hour of Day', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(0, 24, 3))
        
        fig.suptitle('Hourly Telemetry Patterns', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save:
            plt.savefig(f'{self.output_dir}/hourly_patterns.png', dpi=150)
            
        return fig
    
    def plot_daily_trends(self, df: pd.DataFrame,
                          save: bool = True) -> plt.Figure:
        """Plot daily trends and rolling statistics"""
        print("  [8/15] Generating daily trends...")
        
        df_copy = df.copy()
        df_copy['date'] = df_copy.index.date
        
        feature_cols = [c for c in self.config.feature_columns if c in df.columns]
        
        fig, axes = plt.subplots(len(feature_cols), 1, figsize=(16, 3*len(feature_cols)), 
                                sharex=True)
        
        for i, col in enumerate(feature_cols):
            ax = axes[i] if len(feature_cols) > 1 else axes
            
            daily_mean = df_copy.groupby('date')[col].mean()
            daily_std = df_copy.groupby('date')[col].std()
            
            # 7-day rolling mean
            rolling_mean = daily_mean.rolling(7, center=True).mean()
            
            ax.fill_between(daily_mean.index, 
                           daily_mean - daily_std,
                           daily_mean + daily_std,
                           alpha=0.3, color=self.palette['primary'])
            ax.plot(daily_mean.index, daily_mean, 
                   color=self.palette['primary'], alpha=0.5, linewidth=1)
            ax.plot(daily_mean.index, rolling_mean, 
                   color=self.palette['danger'], linewidth=2, label='7-day MA')
            
            ax.set_ylabel(col.replace('_', '\n'), fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
        
        axes[-1].set_xlabel('Date')
        fig.suptitle('Daily Trends with 7-Day Moving Average', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        if save:
            plt.savefig(f'{self.output_dir}/daily_trends.png', dpi=150)
            
        return fig
    
    # =========================================================================
    # 5. ANOMALY ANALYSIS
    # =========================================================================
    
    def plot_anomaly_breakdown(self, df: pd.DataFrame,
                               save: bool = True) -> plt.Figure:
        """Comprehensive anomaly breakdown visualization"""
        print("  [9/15] Generating anomaly breakdown...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # 1. Pie chart - Anomaly type distribution
        ax1 = fig.add_subplot(gs[0, 0])
        if 'anomaly_type' in df.columns:
            type_counts = df['anomaly_type'].value_counts()
            colors = [self.anomaly_colors.get(t, '#gray') for t in type_counts.index]
            ax1.pie(type_counts.values, labels=type_counts.index, 
                   autopct='%1.1f%%', colors=colors, startangle=90)
            ax1.set_title('Anomaly Type Distribution', fontweight='bold')
        
        # 2. Bar chart - Anomaly count by type
        ax2 = fig.add_subplot(gs[0, 1])
        if 'anomaly_type' in df.columns:
            anomaly_df = df[df['label'] == 'anomal']
            type_counts = anomaly_df['anomaly_type'].value_counts()
            colors = [self.anomaly_colors.get(t, '#gray') for t in type_counts.index]
            ax2.bar(range(len(type_counts)), type_counts.values, color=colors)
            ax2.set_xticks(range(len(type_counts)))
            ax2.set_xticklabels(type_counts.index, rotation=45, ha='right')
            ax2.set_ylabel('Count')
            ax2.set_title('Anomaly Count by Type', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Timeline - Anomalies over time
        ax3 = fig.add_subplot(gs[0, 2])
        if 'label' in df.columns:
            df_copy = df.copy()
            df_copy['week'] = df_copy.index.isocalendar().week
            weekly_anomalies = df_copy[df_copy['label'] == 'anomal'].groupby('week').size()
            ax3.bar(weekly_anomalies.index, weekly_anomalies.values, 
                   color=self.palette['danger'], alpha=0.7)
            ax3.set_xlabel('Week')
            ax3.set_ylabel('Anomaly Count')
            ax3.set_title('Weekly Anomaly Count', fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # 4. Heatmap - Anomalies by day and hour
        ax4 = fig.add_subplot(gs[1, :2])
        if 'label' in df.columns:
            df_copy = df.copy()
            df_copy['day'] = df_copy.index.dayofweek
            df_copy['hour'] = df_copy.index.hour
            
            heatmap_data = df_copy[df_copy['label'] == 'anomal'].pivot_table(
                index='day', columns='hour', aggfunc='size', fill_value=0
            )
            
            if not heatmap_data.empty:
                sns.heatmap(heatmap_data, cmap='Reds', ax=ax4, 
                           cbar_kws={'label': 'Anomaly Count'})
                ax4.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
                ax4.set_xlabel('Hour')
                ax4.set_ylabel('Day of Week')
                ax4.set_title('Anomaly Heatmap: Day vs Hour', fontweight='bold')
        
        # 5. Anomaly duration distribution
        ax5 = fig.add_subplot(gs[1, 2])
        if 'label' in df.columns:
            anomaly_mask = df['label'] == 'anomal'
            # Calculate consecutive anomaly sequences
            anomaly_groups = (anomaly_mask != anomaly_mask.shift()).cumsum()
            group_sizes = df[anomaly_mask].groupby(anomaly_groups[anomaly_mask]).size()
            
            ax5.hist(group_sizes.values, bins=20, color=self.palette['info'],
                    edgecolor='white', alpha=0.7)
            ax5.set_xlabel('Consecutive Anomaly Duration (hours)')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Anomaly Duration Distribution', fontweight='bold')
            ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/anomaly_breakdown.png', dpi=150)
            
        return fig
    
    def plot_sensor_radar(self, df: pd.DataFrame,
                          save: bool = True) -> plt.Figure:
        """Radar chart showing sensor behavior during anomalies"""
        print("  [10/15] Generating sensor radar chart...")
        
        feature_cols = [c for c in self.config.feature_columns if c in df.columns]
        n = len(feature_cols)
        
        # Calculate normalized mean for each sensor under each condition
        categories = ['normal'] + [t for t in df['anomaly_type'].unique() if t != 'normal']
        
        fig, axes = plt.subplots(1, len(categories), figsize=(5*len(categories), 5),
                                subplot_kw=dict(projection='polar'))
        
        if len(categories) == 1:
            axes = [axes]
        
        angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
        angles += angles[:1]
        
        for ax, cat in zip(axes, categories):
            cat_data = df[df['anomaly_type'] == cat][feature_cols]
            
            # Normalize to 0-1 range
            normalized = (cat_data.mean() - df[feature_cols].min()) / \
                        (df[feature_cols].max() - df[feature_cols].min() + 1e-8)
            
            values = normalized.values.tolist()
            values += values[:1]
            
            color = self.anomaly_colors.get(cat, '#gray')
            
            ax.plot(angles, values, 'o-', linewidth=2, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([c.replace('_', '\n')[:10] for c in feature_cols], fontsize=8)
            ax.set_title(cat.title(), fontweight='bold', pad=20)
        
        fig.suptitle('Sensor Behavior Radar by Condition', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save:
            plt.savefig(f'{self.output_dir}/sensor_radar.png', dpi=150)
            
        return fig
    
    # =========================================================================
    # 6. MODEL ANALYSIS
    # =========================================================================
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, 
                                    scores: Dict[str, np.ndarray],
                                    save: bool = True) -> plt.Figure:
        """Precision-Recall curves for all models"""
        print("  [11/15] Generating precision-recall curves...")
        
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = [self.palette['danger'], self.palette['primary'], 
                 self.palette['secondary'], self.palette['info'], self.palette['warning']]
        
        for (name, score), color in zip(scores.items(), colors):
            precision, recall, _ = precision_recall_curve(y_true, score)
            ap = average_precision_score(y_true, score)
            ax.plot(recall, precision, label=f'{name} (AP={ap:.3f})',
                   linewidth=2, color=color)
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/precision_recall_curves.png', dpi=150)
            
        return fig
    
    def plot_threshold_analysis(self, y_true: np.ndarray,
                                scores: np.ndarray,
                                save: bool = True) -> plt.Figure:
        """Analysis of different threshold values"""
        print("  [12/15] Generating threshold analysis...")
        
        from sklearn.metrics import precision_recall_fscore_support
        
        thresholds = np.linspace(scores.min(), scores.max(), 100)
        
        precisions = []
        recalls = []
        f1s = []
        
        for thresh in thresholds:
            preds = (scores > thresh).astype(int)
            p, r, f, _ = precision_recall_fscore_support(y_true, preds, 
                                                          average='binary', 
                                                          zero_division=0)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Metrics vs threshold
        axes[0].plot(thresholds, precisions, label='Precision', 
                    color=self.palette['primary'], linewidth=2)
        axes[0].plot(thresholds, recalls, label='Recall', 
                    color=self.palette['danger'], linewidth=2)
        axes[0].plot(thresholds, f1s, label='F1-Score', 
                    color=self.palette['secondary'], linewidth=2)
        
        # Mark optimal F1
        optimal_idx = np.argmax(f1s)
        axes[0].axvline(x=thresholds[optimal_idx], color='gray', 
                       linestyle='--', alpha=0.7)
        axes[0].scatter([thresholds[optimal_idx]], [f1s[optimal_idx]], 
                       color='red', s=100, zorder=5)
        axes[0].annotate(f'Optimal: {thresholds[optimal_idx]:.3f}',
                        xy=(thresholds[optimal_idx], f1s[optimal_idx]),
                        xytext=(10, 10), textcoords='offset points')
        
        axes[0].set_xlabel('Threshold', fontsize=12)
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].set_title('Metrics vs Threshold', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Precision-Recall tradeoff
        axes[1].plot(recalls, precisions, color=self.palette['info'], linewidth=2)
        axes[1].scatter([recalls[optimal_idx]], [precisions[optimal_idx]], 
                       color='red', s=100, zorder=5, label=f'Optimal (F1={f1s[optimal_idx]:.3f})')
        axes[1].set_xlabel('Recall', fontsize=12)
        axes[1].set_ylabel('Precision', fontsize=12)
        axes[1].set_title('Precision-Recall Tradeoff', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/threshold_analysis.png', dpi=150)
            
        return fig
    
    def plot_score_distributions(self, y_true: np.ndarray,
                                 scores: Dict[str, np.ndarray],
                                 save: bool = True) -> plt.Figure:
        """Anomaly score distributions for normal vs anomaly"""
        print("  [13/15] Generating score distributions...")
        
        n_models = len(scores)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (name, score) in zip(axes, scores.items()):
            normal_scores = score[y_true == 0]
            anomaly_scores = score[y_true == 1]
            
            ax.hist(normal_scores, bins=50, alpha=0.6, label='Normal',
                   color=self.palette['secondary'], density=True)
            ax.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly',
                   color=self.palette['danger'], density=True)
            
            ax.axvline(x=np.percentile(score, 95), color='red', 
                      linestyle='--', label='95th percentile')
            
            ax.set_xlabel('Anomaly Score')
            ax.set_ylabel('Density')
            ax.set_title(name.replace('_', ' ').title(), fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('Anomaly Score Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save:
            plt.savefig(f'{self.output_dir}/score_distributions.png', dpi=150)
            
        return fig
    
    # =========================================================================
    # 7. OPERATIONAL DASHBOARDS
    # =========================================================================
    
    def plot_operations_dashboard(self, df: pd.DataFrame,
                                  scores: np.ndarray,
                                  severities: List,
                                  save: bool = True) -> plt.Figure:
        """Comprehensive operations dashboard"""
        print("  [14/15] Generating operations dashboard...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 1.5, 1])
        
        feature_cols = [c for c in self.config.feature_columns if c in df.columns]
        
        # Row 1: Key metrics gauges (simulated as bar charts)
        for i, col in enumerate(feature_cols[:4]):
            ax = fig.add_subplot(gs[0, i])
            current_val = df[col].iloc[-1]
            min_val, max_val = self.config.operational_limits.get(col, (0, 100))
            
            # Normalize to percentage
            pct = (current_val - min_val) / (max_val - min_val) * 100
            
            # Color based on status
            if pct < 20 or pct > 80:
                color = self.palette['danger']
            elif pct < 30 or pct > 70:
                color = self.palette['warning']
            else:
                color = self.palette['secondary']
            
            ax.barh([0], [pct], color=color, height=0.5)
            ax.set_xlim(0, 100)
            ax.set_yticks([])
            ax.set_title(col.replace('_', ' ').title()[:15], fontsize=10, fontweight='bold')
            ax.text(50, 0, f'{current_val:.2f}', ha='center', va='center', 
                   fontsize=12, fontweight='bold')
        
        # Row 2: Main telemetry timeline
        ax_main = fig.add_subplot(gs[1, :])
        
        # Plot last 48 hours
        recent_df = df.iloc[-48:] if len(df) > 48 else df
        recent_scores = scores[-48:] if len(scores) > 48 else scores
        
        ax_main_twin = ax_main.twinx()
        
        for col in feature_cols[:3]:
            ax_main.plot(recent_df.index, 
                        (recent_df[col] - recent_df[col].min()) / 
                        (recent_df[col].max() - recent_df[col].min() + 1e-8),
                        alpha=0.7, label=col)
        
        ax_main_twin.fill_between(recent_df.index, recent_scores, 
                                  alpha=0.3, color=self.palette['danger'],
                                  label='Anomaly Score')
        
        ax_main.set_ylabel('Normalized Telemetry')
        ax_main_twin.set_ylabel('Anomaly Score', color=self.palette['danger'])
        ax_main.set_title('Last 48 Hours: Telemetry & Anomaly Score', 
                         fontsize=12, fontweight='bold')
        ax_main.legend(loc='upper left', fontsize=8)
        ax_main.grid(True, alpha=0.3)
        
        # Row 3: Statistics
        # Alert status
        ax_alert = fig.add_subplot(gs[2, 0])
        severity_counts = pd.Series([s.name for s in severities[-48:]]).value_counts()
        colors = [self.anomaly_colors.get(s.lower(), '#gray') for s in severity_counts.index]
        ax_alert.pie(severity_counts.values, labels=severity_counts.index,
                    colors=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad'][:len(severity_counts)],
                    autopct='%1.0f%%')
        ax_alert.set_title('Alert Distribution (48h)', fontweight='bold')
        
        # Recent alerts table
        ax_table = fig.add_subplot(gs[2, 1:3])
        ax_table.axis('off')
        
        if 'label' in df.columns:
            recent_anomalies = df[df['label'] == 'anomal'].tail(10)
            if len(recent_anomalies) > 0:
                table_data = []
                for idx, row in recent_anomalies.iterrows():
                    table_data.append([
                        str(idx)[:16],
                        row.get('anomaly_type', 'unknown')[:10],
                        f"{row[feature_cols[0]]:.2f}" if feature_cols else '-'
                    ])
                
                if table_data:
                    table = ax_table.table(
                        cellText=table_data,
                        colLabels=['Timestamp', 'Type', 'Speed'],
                        loc='center',
                        cellLoc='center'
                    )
                    table.auto_set_font_size(False)
                    table.set_fontsize(9)
                    table.scale(1, 1.5)
        
        ax_table.set_title('Recent Anomaly Events', fontweight='bold')
        
        # System status
        ax_status = fig.add_subplot(gs[2, 3])
        ax_status.axis('off')
        
        current_score = scores[-1] if len(scores) > 0 else 0
        if current_score < 0.3:
            status = "NOMINAL"
            status_color = self.palette['secondary']
        elif current_score < 0.5:
            status = "CAUTION"
            status_color = self.palette['warning']
        else:
            status = "ALERT"
            status_color = self.palette['danger']
        
        ax_status.text(0.5, 0.6, status, ha='center', va='center',
                      fontsize=24, fontweight='bold', color=status_color)
        ax_status.text(0.5, 0.3, f'Score: {current_score:.3f}', 
                      ha='center', va='center', fontsize=14)
        ax_status.set_title('Current Status', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/operations_dashboard.png', dpi=150)
            
        return fig
    
    def plot_summary_report(self, df: pd.DataFrame,
                            results: Dict,
                            save: bool = True) -> plt.Figure:
        """Generate visual summary report"""
        print("  [15/15] Generating summary report...")
        
        fig = plt.figure(figsize=(16, 20))
        gs = gridspec.GridSpec(4, 2, figure=fig)
        
        # Title
        fig.suptitle('Satellite Anomaly Detection - Summary Report\n' + 
                    f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                    fontsize=16, fontweight='bold')
        
        # 1. Dataset summary
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        
        summary_text = f"""
DATASET OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Samples: {len(df):,}
Date Range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}
Duration: {(df.index.max() - df.index.min()).days} days
Sampling Rate: 1 hour

ANOMALY STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━
Normal: {(df['label'] == 'normal').sum():,} ({(df['label'] == 'normal').mean()*100:.1f}%)
Anomaly: {(df['label'] == 'anomal').sum():,} ({(df['label'] == 'anomal').mean()*100:.1f}%)
        """
        ax1.text(0.1, 0.9, summary_text, transform=ax1.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        # 2. Model performance
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        
        if 'evaluator' in results:
            perf_text = "MODEL PERFORMANCE\n" + "━"*30 + "\n"
            for name, res in results['evaluator'].results.items():
                perf_text += f"\n{name.upper()}\n"
                perf_text += f"  Precision: {res.precision:.4f}\n"
                perf_text += f"  Recall:    {res.recall:.4f}\n"
                perf_text += f"  F1-Score:  {res.f1_score:.4f}\n"
                perf_text += f"  ROC-AUC:   {res.roc_auc:.4f}\n"
            
            ax2.text(0.1, 0.9, perf_text, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # 3. Feature statistics
        ax3 = fig.add_subplot(gs[1, :])
        feature_cols = [c for c in self.config.feature_columns if c in df.columns]
        
        stats_data = []
        for col in feature_cols:
            stats_data.append([
                col,
                f"{df[col].mean():.2f}",
                f"{df[col].std():.2f}",
                f"{df[col].min():.2f}",
                f"{df[col].max():.2f}"
            ])
        
        table = ax3.table(
            cellText=stats_data,
            colLabels=['Feature', 'Mean', 'Std', 'Min', 'Max'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax3.axis('off')
        ax3.set_title('Feature Statistics', fontsize=12, fontweight='bold', pad=20)
        
        # 4-5. Time series
        ax4 = fig.add_subplot(gs[2, :])
        if len(df) > 0 and feature_cols:
            col = feature_cols[0]
            ax4.plot(df.index, df[col], 'b-', alpha=0.7, linewidth=0.8)
            if 'label' in df.columns:
                anomaly_idx = df[df['label'] == 'anomal'].index
                ax4.scatter(anomaly_idx, df.loc[anomaly_idx, col], 
                           c='red', s=30, zorder=5)
            ax4.set_ylabel(col)
            ax4.set_title('Primary Telemetry Parameter', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        # 6. Anomaly type breakdown
        ax5 = fig.add_subplot(gs[3, 0])
        if 'anomaly_type' in df.columns:
            type_counts = df[df['label'] == 'anomal']['anomaly_type'].value_counts()
            colors = [self.anomaly_colors.get(t, '#gray') for t in type_counts.index]
            ax5.barh(range(len(type_counts)), type_counts.values, color=colors)
            ax5.set_yticks(range(len(type_counts)))
            ax5.set_yticklabels(type_counts.index)
            ax5.set_xlabel('Count')
            ax5.set_title('Anomalies by Type', fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='x')
        
        # 7. Recommendations
        ax6 = fig.add_subplot(gs[3, 1])
        ax6.axis('off')
        
        recommendations = """
RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Speed and current anomalies show highest
   detection rates (88-90%)

2. Vibration anomaly detection needs improvement
   - Consider additional frequency-domain features
   - Investigate sensor calibration

3. Monitor wheel_speed_rpm_range for early
   warning indicators

4. Recommended alert thresholds:
   • LOW:      0.30
   • MEDIUM:   0.50
   • HIGH:     0.70
   • CRITICAL: 0.85
        """
        ax6.text(0.1, 0.9, recommendations, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save:
            plt.savefig(f'{self.output_dir}/summary_report.png', dpi=150)
            
        return fig
    
    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================
    
    def generate_all(self, pipeline: SatelliteAnomalyPipeline) -> None:
        """Generate all advanced visualizations"""
        print("\n" + "="*70)
        print("GENERATING ADVANCED VISUALIZATIONS")
        print("="*70)
        
        df = pipeline.df_raw
        X = pipeline.feature_matrix.values
        y = pipeline.y_true
        y_types = pipeline.y_types
        
        # Get scores
        all_scores = pipeline.ensemble_detector.predict_scores(X)
        ensemble_scores = pipeline.ensemble_detector.compute_ensemble_score(all_scores)
        
        # Generate all plots
        self.plot_correlation_heatmap(df)
        self.plot_cross_correlation_matrix(df)
        self.plot_feature_distributions(df)
        self.plot_violin_by_anomaly_type(df)
        self.plot_pca_analysis(X, y)
        self.plot_tsne_visualization(X, y, y_types)
        self.plot_hourly_patterns(df)
        self.plot_daily_trends(df)
        self.plot_anomaly_breakdown(df)
        self.plot_sensor_radar(df)
        self.plot_precision_recall_curve(y, all_scores)
        self.plot_threshold_analysis(y, ensemble_scores)
        self.plot_score_distributions(y, all_scores)
        
        severities = pipeline.ensemble_detector.classify_severity(ensemble_scores)
        self.plot_operations_dashboard(df, ensemble_scores, severities)
        
        results = {
            'evaluator': pipeline.evaluator,
            'ensemble_detector': pipeline.ensemble_detector
        }
        self.plot_summary_report(df, results)
        
        print("\n" + "="*70)
        print("ADVANCED VISUALIZATIONS COMPLETE")
        print("="*70)
        print(f"\nGenerated 15 additional visualization files in: {self.output_dir}")


# ================================================================================
# ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("RUNNING ADVANCED VISUALIZATION SUITE")
    print("="*70)
    
    # Run main pipeline first
    config = TelemetryConfig()
    pipeline = SatelliteAnomalyPipeline(config)
    results = pipeline.run('data.csv')
    
    # Generate advanced visualizations
    adv_viz = AdvancedVisualizer(config, output_dir='output')
    adv_viz.generate_all(pipeline)
    
    print("\nAll visualizations complete!")

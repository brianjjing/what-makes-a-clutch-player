"""
visualize_clutch.py
-------------------
Creates 3 core visualizations for NBA clutch score analysis.

Usage:
    python visualize_clutch.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuration
DATA_FILE = Path("data/outputs/clutch_scores_all_seasons.csv")
VIZ_DIR = Path("data/visualizations")
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_data(min_clutch_fga=0.5):
    """Load and filter data with minimum clutch FGA to avoid small samples."""
    df = pd.read_csv(DATA_FILE)
    df_filtered = df[df['FGA_CLU'] >= min_clutch_fga].copy()
    
    print(f"Loaded {len(df)} total player-seasons")
    print(f"Filtered to {len(df_filtered)} with >= {min_clutch_fga} clutch FGA/game")
    print(f"Regular season minimum: 10 min/game\n")
    
    return df_filtered

def plot_distribution(df):
    """Distribution of clutch scores - shows the overall landscape."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Histogram with stats
    ax.hist(df['CLUTCH_SCORE'].dropna(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(df['CLUTCH_SCORE'].median(), color='red', linestyle='--', linewidth=2.5, 
               label=f'Median: {df["CLUTCH_SCORE"].median():.3f}')
    ax.axvline(df['CLUTCH_SCORE'].mean(), color='orange', linestyle='--', linewidth=2.5, 
               label=f'Mean: {df["CLUTCH_SCORE"].mean():.3f}')
    
    ax.set_xlabel('Clutch Score', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Players', fontsize=13, fontweight='bold')
    ax.set_title('Distribution of NBA Clutch Scores (2020-2025)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'clutch_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: clutch_distribution.png")
    plt.close()

def plot_regular_vs_clutch(df):
    """THE KEY PLOT: Regular season vs clutch efficiency scatter."""
    fig, ax = plt.subplots(figsize=(13, 10))
    
    # Calculate improvement
    df['IMPROVEMENT'] = df['TS_PCT_CLU'] - df['TS_PCT_REG']
    
    # Scatter with color gradient
    scatter = ax.scatter(df['TS_PCT_REG'], df['TS_PCT_CLU'], 
                        c=df['IMPROVEMENT'], cmap='RdYlGn', vmin=-0.3, vmax=0.3,
                        s=120, alpha=0.7, edgecolors='black', linewidth=0.7)
    
    # Reference line (no change)
    lims = [min(df['TS_PCT_REG'].min(), df['TS_PCT_CLU'].min()), 
            max(df['TS_PCT_REG'].max(), df['TS_PCT_CLU'].max())]
    ax.plot(lims, lims, 'k--', alpha=0.6, linewidth=2.5, label='No Change (Equal Performance)')
    
    ax.set_xlabel('Regular Season TS%', fontsize=13, fontweight='bold')
    ax.set_ylabel('Clutch TS%', fontsize=13, fontweight='bold')
    ax.set_title('Who Actually Performs Better in the Clutch?\n(Points above line = Better in clutch)', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Clutch Improvement', fontsize=11, fontweight='bold')
    
    # Add stats text box
    improved = (df['IMPROVEMENT'] > 0).sum()
    total = len(df['IMPROVEMENT'].dropna())
    pct = 100 * improved / total
    textstr = f'Players who improve in clutch: {improved}/{total} ({pct:.1f}%)'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Label notable players (top improvers and worst decliners)
    df_labeled = df.dropna(subset=['IMPROVEMENT', 'TS_PCT_REG', 'TS_PCT_CLU'])
    
    # Top 10 clutch improvers
    top_improvers = df_labeled.nlargest(10, 'IMPROVEMENT')
    for _, row in top_improvers.iterrows():
        ax.annotate(f"{row['PLAYER_NAME']}\n({row['SEASON']})", 
                   xy=(row['TS_PCT_REG'], row['TS_PCT_CLU']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7, edgecolor='black'))
    
    # Top 10 worst decliners
    worst_decliners = df_labeled.nsmallest(10, 'IMPROVEMENT')
    for _, row in worst_decliners.iterrows():
        ax.annotate(f"{row['PLAYER_NAME']}\n({row['SEASON']})", 
                   xy=(row['TS_PCT_REG'], row['TS_PCT_CLU']),
                   xytext=(5, -5), textcoords='offset points',
                   fontsize=8, alpha=0.8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7, edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'regular_vs_clutch.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: regular_vs_clutch.png")
    plt.close()

def plot_top_performers(df, n=15):
    """Top N clutch performers bar chart."""
    top = df.nlargest(n, 'CLUTCH_SCORE')[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'SEASON', 'CLUTCH_SCORE', 'FGA_CLU']]
    top['LABEL'] = top['PLAYER_NAME'] + ' (' + top['SEASON'] + ')'
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Create bars with gradient color
    bars = ax.barh(range(len(top)), top['CLUTCH_SCORE'], color='steelblue', edgecolor='black', linewidth=1.2)
    colors = plt.cm.RdYlGn(np.linspace(0.4, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['LABEL'].values)
    ax.set_xlabel('Clutch Score', fontsize=13, fontweight='bold')
    ax.set_title(f'Top {n} Clutch Performers (2020-2025)', fontsize=15, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add score annotations
    for i, (score, fga) in enumerate(zip(top['CLUTCH_SCORE'], top['FGA_CLU'])):
        ax.text(score + 0.015, i, f'{score:.3f}', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f'top_{n}_clutch.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: top_{n}_clutch.png")
    plt.close()

def print_stats(df):
    """Print key statistics."""
    print("="*60)
    print("KEY STATISTICS")
    print("="*60)
    print(f"Mean Clutch Score:       {df['CLUTCH_SCORE'].mean():.3f}")
    print(f"Median Clutch Score:     {df['CLUTCH_SCORE'].median():.3f}")
    print(f"Best:                    {df['CLUTCH_SCORE'].max():.3f}")
    print(f"Worst:                   {df['CLUTCH_SCORE'].min():.3f}")
    
    improvement = df['TS_PCT_CLU'] - df['TS_PCT_REG']
    improved_pct = 100 * (improvement > 0).sum() / len(improvement)
    print(f"\nPlayers who improve in clutch: {improved_pct:.1f}%")
    print(f"Average improvement:           {improvement.mean():.3f}")
    print("="*60 + "\n")

def main():
    print("\n" + "="*60)
    print("NBA CLUTCH SCORE VISUALIZATIONS")
    print("="*60 + "\n")
    
    # Load data
    df = load_data(min_clutch_fga=0.5)
    
    # Stats
    print_stats(df)
    
    # Generate plots
    print("Generating visualizations...\n")
    plot_distribution(df)
    plot_regular_vs_clutch(df)
    plot_top_performers(df, n=15)
    
    print("\n" + "="*60)
    print(f"✅ Complete! All visualizations saved to: {VIZ_DIR}/")
    print("="*60)

if __name__ == "__main__":
    main()
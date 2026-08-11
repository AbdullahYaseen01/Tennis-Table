from __future__ import annotations

BG_DEEP = "#0b0f14"
BG_MAIN = "#111820"
BG_CARD = "#1a2332"
BG_CARD_HOVER = "#222d3f"
BG_ELEVATED = "#252f42"

ACCENT_LIME = "#c8f135"      
ACCENT_TEAL = "#00d4aa"
ACCENT_CORAL = "#ff6b4a"
ACCENT_SKY = "#4dabf7"
ACCENT_GOLD = "#fbbf24"
ACCENT_VIOLET = "#a78bfa"

TEXT_PRIMARY = "#f1f5f9"
TEXT_SECONDARY = "#94a3b8"
TEXT_MUTED = "#64748b"

BORDER = "rgba(255, 255, 255, 0.08)"
BORDER_ACCENT = "rgba(200, 241, 53, 0.35)"

CHART_COMPRESSION = ACCENT_CORAL
CHART_RECOVERY = ACCENT_TEAL
CHART_ZONE_OK = ACCENT_TEAL
CHART_ZONE_FLAG = ACCENT_CORAL
CHART_FATIGUE_1 = ACCENT_CORAL
CHART_FATIGUE_2 = ACCENT_SKY
CHART_FATIGUE_3 = ACCENT_LIME

APP_STYLESHEET = f"""
QMainWindow, QWidget {{ 
    background-color: {BG_MAIN};
    color: {TEXT_PRIMARY};
    font-family: "Segoe UI", "SF Pro Display", sans-serif;
    font-size: 13px;
}} 

QTabWidget::pane {{ 
    border: 1px solid {BORDER};
    border-radius: 0;
    background: {BG_MAIN};
    top: -1px;
}} 

QTabBar::tab {{ 
    background: {BG_DEEP};
    color: {TEXT_SECONDARY};
    padding: 12px 28px;
    margin-right: 4px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: 600;
    font-size: 13px;
}} 

QTabBar::tab:selected {{ 
    background: {BG_CARD};
    color: {ACCENT_LIME};
    border-bottom: 2px solid {ACCENT_LIME};
}} 

QTabBar::tab:hover:!selected {{ 
    background: {BG_CARD_HOVER};
    color: {TEXT_PRIMARY};
}} 

QScrollBar:vertical {{ 
    background: {BG_DEEP};
    width: 10px;
    border-radius: 5px;
}} 
QScrollBar::handle:vertical {{ 
    background: {BG_ELEVATED};
    border-radius: 5px;
    min-height: 30px;
}} 
QScrollBar::handle:vertical:hover {{ 
    background: {TEXT_MUTED};
}} 

QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{ 
    background: {BG_ELEVATED};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 8px 12px;
    min-height: 20px;
}} 
QComboBox:hover, QLineEdit:hover {{ 
    border-color: {BORDER_ACCENT};
}} 
QComboBox::drop-down {{ 
    border: none;
    width: 24px;
}} 
QComboBox QAbstractItemView {{ 
    background: {BG_CARD};
    color: {TEXT_PRIMARY};
    selection-background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
}} 

QPushButton {{ 
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {ACCENT_LIME}, stop:1 #9ae234);
    color: #0b0f14;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 700;
    font-size: 13px;
}} 
QPushButton:hover {{ 
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #d4f542, stop:1 {ACCENT_LIME});
}} 
QPushButton:pressed {{ 
    background: #9ae234;
}} 

QPushButton[class="secondary"] {{ 
    background: {BG_ELEVATED};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
}} 
QPushButton[class="secondary"]:hover {{ 
    border-color: {BORDER_ACCENT};
    background: {BG_CARD_HOVER};
}} 

QSplitter::handle {{ 
    background: {BORDER};
    width: 2px;
}} 
"""

DASHBOARD_STYLESHEET = f"""
#dashboardRoot {{ 
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 {BG_DEEP}, stop:0.5 {BG_MAIN}, stop:1 #0d1520);
}} 

#heroBanner {{ 
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1a2838, stop:0.4 #1e3348, stop:1 #162636);
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 4px;
}} 

#heroTitle {{ 
    font-size: 26px;
    font-weight: 800;
    color: {TEXT_PRIMARY};
    letter-spacing: -0.5px;
}} 

#heroSubtitle {{ 
    font-size: 13px;
    color: {TEXT_SECONDARY};
    font-weight: 400;
}} 

#filterBar {{ 
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 4px 8px;
}} 

#filterBar QLabel {{ 
    color: {TEXT_SECONDARY};
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}} 

#statCard {{ 
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {BG_ELEVATED}, stop:1 {BG_CARD});
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 16px;
    min-width: 140px;
}} 

#statCard[accent="lime"] {{ 
    border-left: 4px solid {ACCENT_LIME};
}} 
#statCard[accent="teal"] {{ 
    border-left: 4px solid {ACCENT_TEAL};
}} 
#statCard[accent="coral"] {{ 
    border-left: 4px solid {ACCENT_CORAL};
}} 
#statCard[accent="sky"] {{ 
    border-left: 4px solid {ACCENT_SKY};
}} 

#statCardTitle {{ 
    color: {TEXT_MUTED};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}} 

#statCardValue {{ 
    color: {TEXT_PRIMARY};
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -1px;
}} 

#statCardSub {{ 
    color: {TEXT_SECONDARY};
    font-size: 11px;
}} 

#chartCard {{ 
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 14px;
}} 

#chartCardTitle {{ 
    color: {TEXT_PRIMARY};
    font-size: 14px;
    font-weight: 700;
    padding: 4px 0;
}} 

#chartCardSubtitle {{ 
    color: {TEXT_MUTED};
    font-size: 11px;
}} 

#runTable {{ 
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 14px;
    gridline-color: {BORDER};
    selection-background-color: rgba(200, 241, 53, 0.15);
    selection-color: {TEXT_PRIMARY};
    outline: none;
}} 

#runTable QHeaderView::section {{ 
    background: {BG_ELEVATED};
    color: {TEXT_SECONDARY};
    font-weight: 700;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 12px 8px;
    border: none;
    border-bottom: 2px solid {ACCENT_LIME};
}} 

#runTable::item {{ 
    padding: 10px 8px;
    border-bottom: 1px solid {BORDER};
}} 

#runTable::item:selected {{ 
    background: rgba(200, 241, 53, 0.12);
}} 

#runTable::item:hover {{ 
    background: rgba(255, 255, 255, 0.04);
}} 

#runTable::item:alternate {{ 
    background: rgba(255, 255, 255, 0.02);
}} 

#detailHeader {{ 
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 rgba(200, 241, 53, 0.12), stop:1 transparent);
    border: 1px solid {BORDER_ACCENT};
    border-radius: 12px;
    padding: 14px 18px;
}} 

#detailHeaderTitle {{ 
    font-size: 16px;
    font-weight: 700;
    color: {ACCENT_LIME};
}} 

#detailHeaderMeta {{ 
    color: {TEXT_SECONDARY};
    font-size: 12px;
}} 

#sectionLabel {{ 
    font-size: 15px;
    font-weight: 800;
    color: {TEXT_PRIMARY};
    letter-spacing: -0.3px;
}} 

#sectionHint {{ 
    color: {TEXT_MUTED};
    font-size: 12px;
}} 

#fatigueSection {{ 
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {BG_CARD}, stop:1 #1a2a3a);
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 8px;
}} 
"""

SIMPLE_DASHBOARD_STYLESHEET = f"""
#dashTitle {{ 
    font-size: 18px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
}} 

#dashSummary {{ 
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 12px 14px;
    color: {TEXT_SECONDARY};
    font-size: 13px;
}} 

#simpleTable {{ 
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    gridline-color: {BORDER};
    selection-background-color: rgba(200, 241, 53, 0.12);
}} 

#simpleTable QHeaderView::section {{ 
    background: {BG_ELEVATED};
    color: {TEXT_SECONDARY};
    font-weight: 600;
    padding: 10px 6px;
    border: none;
    border-bottom: 1px solid {BORDER};
}} 

#trendBox {{ 
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
}} 
"""

def style_plot(widget, *, title: str = "") -> None:
    
    widget.setBackground(BG_ELEVATED)
    widget.getPlotItem().getViewBox().setBackgroundColor(BG_CARD)
    for axis_name in ("left", "bottom"):
        axis = widget.getPlotItem().getAxis(axis_name)
        axis.setPen(TEXT_MUTED)
        axis.setTextPen(TEXT_SECONDARY)
    if title:
        widget.setTitle(title, color=TEXT_PRIMARY, size="11pt")
    widget.showGrid(x=True, y=True, alpha=0.15)

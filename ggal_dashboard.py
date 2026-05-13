
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG PÁGINA
# ============================================================
st.set_page_config(
    page_title="GGAL Options Scanner",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .alert-card {
        background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
        border: 1px solid #30363d; border-radius: 12px;
        padding: 18px; margin-bottom: 14px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .card-entry   { border-left: 4px solid #2ea043; }
    .card-warning { border-left: 4px solid #d29922; }
    .card-danger  { border-left: 4px solid #f85149; }
    .card-info    { border-left: 4px solid #388bfd; }
    .card-ratio   { border-left: 4px solid #bc8cff; }
    .alert-title  { font-size:15px; font-weight:700; margin-bottom:10px; }
    .alert-detail {
        font-size:12px; color:#8b949e; margin-top:8px;
        font-style:italic; background:#0d1117;
        padding:8px 12px; border-radius:6px;
    }
    .metric-box {
        background:#1c2128; border:1px solid #30363d;
        border-radius:8px; padding:14px; text-align:center;
        margin-bottom:10px;
    }
    .metric-value { font-size:22px; font-weight:700; color:#58a6ff; }
    .metric-label { font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:1px; }
    .tag { display:inline-block; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; margin-right:6px; }
    .tag-call    { background:#1f4e2e; color:#2ea043; }
    .tag-put     { background:#4e1f1f; color:#f85149; }
    .tag-neutral { background:#1f3a4e; color:#388bfd; }
    .timestamp   { font-size:11px; color:#484f58; text-align:right; margin-top:6px; }
    .pct-bar { height:6px; border-radius:3px; margin:4px 0; }
    h1,h2,h3 { color:#e6edf3; }
    .stButton>button {
        background:#21262d; color:#e6edf3;
        border:1px solid #30363d; border-radius:8px;
        width:100%;
    }
    .stButton>button:hover { background:#30363d; border-color:#58a6ff; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# BLACK-SCHOLES
# ============================================================
class BS:
    @staticmethod
    def d1(S,K,T,r,v):
        if T<=0 or v<=0: return 0
        return (np.log(S/K)+(r+.5*v**2)*T)/(v*np.sqrt(T))

    @staticmethod
    def call(S,K,T,r,v):
        if T<=0: return max(S-K,0)
        d1=BS.d1(S,K,T,r,v); d2=d1-v*np.sqrt(T)
        return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)

    @staticmethod
    def put(S,K,T,r,v):
        if T<=0: return max(K-S,0)
        d1=BS.d1(S,K,T,r,v); d2=d1-v*np.sqrt(T)
        return K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)

    @staticmethod
    def iv(price,S,K,T,r,flag='call'):
        try:
            f = (lambda v: BS.call(S,K,T,r,v)-price) if flag=='call' \
                else (lambda v: BS.put(S,K,T,r,v)-price)
            intr = max(S-K,0) if flag=='call' else max(K-S,0)
            if price <= intr+1e-4: return 0.001
            return brentq(f,1e-6,9.0,xtol=1e-5,maxiter=200)
        except: return np.nan

    @staticmethod
    def delta(S,K,T,r,v,flag='call'):
        if T<=0 or v<=0: return 0
        d1=BS.d1(S,K,T,r,v)
        return norm.cdf(d1) if flag=='call' else norm.cdf(d1)-1

    @staticmethod
    def gamma(S,K,T,r,v):
        if T<=0 or v<=0: return 0
        d1=BS.d1(S,K,T,r,v)
        return norm.pdf(d1)/(S*v*np.sqrt(T))

    @staticmethod
    def theta(S,K,T,r,v,flag='call'):
        if T<=0 or v<=0: return 0
        d1=BS.d1(S,K,T,r,v); d2=d1-v*np.sqrt(T)
        t1=-(S*norm.pdf(d1)*v)/(2*np.sqrt(T))
        if flag=='call': return (t1-r*K*np.exp(-r*T)*norm.cdf(d2))/365
        return (t1+r*K*np.exp(-r*T)*norm.cdf(-d2))/365

    @staticmethod
    def vega(S,K,T,r,v):
        if T<=0 or v<=0: return 0
        d1=BS.d1(S,K,T,r,v)
        return S*norm.pdf(d1)*np.sqrt(T)/100


# ============================================================
# FETCHER DE DATOS REALES - GGAL (BYMA vía API pública)
# ============================================================
class GGALDataFetcher:
    """
    Obtiene datos reales de opciones de GGAL desde:
    1. API de Invertir Online (IOL) - pública sin auth para cotizaciones
    2. API de Argentina Datos / BYMA Data
    3. Fallback: Yahoo Finance para spot price
    """

    HEADERS = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }

    # Strikes típicos de GGAL en BYMA (se actualizan con los reales)
    STRIKES_REF = [
        4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000
    ]

    @staticmethod
    @st.cache_data(ttl=60)   # cache 60 segundos
    def get_spot():
        """Precio spot de GGAL desde múltiples fuentes"""
        # Fuente 1: Yahoo Finance (GGAL en NYSE en USD, convertimos)
        try:
            url = "https://query1.finance.yahoo.com/v8/finance/chart/GGAL.BA"
            r = requests.get(url, headers=GGALDataFetcher.HEADERS, timeout=8)
            data = r.json()
            price = data['chart']['result'][0]['meta']['regularMarketPrice']
            if price and price > 0:
                return float(price)
        except: pass

        # Fuente 2: BYMA vía API pública (sin auth)
        try:
            url = "https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free/bnown/seriesHistoricas/GGAL"
            r = requests.get(url, headers=GGALDataFetcher.HEADERS, timeout=8)
            data = r.json()
            if data:
                price = data[0].get('trade', data[0].get('c', 0))
                if price and float(price) > 0:
                    return float(price)
        except: pass

        # Fuente 3: IOL scraping básico
        try:
            url = "https://api.invertironline.com/api/v2/Cotizacion/BCBA/GGAL/Acciones"
            r = requests.get(url, headers=GGALDataFetcher.HEADERS, timeout=8)
            data = r.json()
            price = data.get('ultimoPrecio', data.get('ultimo', 0))
            if price and float(price) > 0:
                return float(price)
        except: pass

        return None   # sin datos

    @staticmethod
    @st.cache_data(ttl=60)
    def get_opciones_byma(spot):
        """
        Descarga la cadena de opciones de GGAL desde BYMA Data API pública.
        Endpoint abierto: no requiere token.
        """
        opciones = []

        # ── BYMA endpoint de opciones ──────────────────────────────────
        try:
            url = ("https://open.bymadata.com.ar/vanoms-be-core/rest/api/"
                   "bymadata/free/opciones")
            params = {"subyacente": "GGAL"}
            r = requests.get(url, params=params,
                             headers=GGALDataFetcher.HEADERS, timeout=10)
            items = r.json()

            for item in items:
                try:
                    ticker  = item.get('symbol','').upper()
                    precio  = float(item.get('trade', item.get('c', 0)) or 0)
                    vol_op  = int(item.get('volumenNominal',
                                          item.get('v', 0)) or 0)
                    oi      = int(item.get('openInterest',
                                          item.get('oi', 0)) or 0)

                    if precio <= 0:
                        continue

                    # Parsear ticker: GFGCmmmmYY / GFGVmmmmYY
                    # C = call, V = put (nomenclatura BYMA)
                    flag, strike, vto = GGALDataFetcher._parse_ticker(ticker)
                    if flag is None:
                        continue

                    dias = (vto - datetime.today()).days
                    if dias < 1:
                        continue

                    opciones.append({
                        'ticker' : ticker,
                        'tipo'   : flag,
                        'strike' : strike,
                        'precio' : precio,
                        'vol_op' : vol_op,
                        'oi'     : oi,
                        'dias'   : dias,
                        'vto'    : vto.strftime('%d/%m/%Y'),
                    })
                except:
                    continue

        except Exception as e:
            st.warning(f"⚠️ BYMA API: {e}")

        # ── Fallback: construir cadena sintética con BS si no hay datos ──
        if not opciones and spot:
            opciones = GGALDataFetcher._cadena_sintetica(spot)

        return pd.DataFrame(opciones)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_ticker(ticker: str):
        """
        Parsea tickers BYMA de opciones sobre GGAL.
        Formatos conocidos:
          GFGC6000OC  →  GGAL CALL 6000 vto Oct
          GFGV5500SE  →  GGAL PUT  5500 vto Sep
        """
        import re
        # Patrón: GFG + C/V + strike(4-5 dígitos) + mes(2 letras)
        m = re.match(r'GFG([CV])(\d{3,5})([A-Z]{2})', ticker)
        if not m:
            return None, None, None

        flag   = 'call' if m.group(1) == 'C' else 'put'
        strike = float(m.group(2))
        mes_cod = m.group(3)

        # Mapeo de códigos de mes BYMA → número de mes
        MESES = {
            'EN':1,'FE':2,'MR':3,'AB':4,'MY':5,'JN':6,
            'JL':7,'AG':8,'SE':9,'OC':10,'NO':11,'DI':12
        }
        mes_num = MESES.get(mes_cod.upper())
        if mes_num is None:
            return None, None, None

        # Tercer viernes del mes (vencimiento estándar BYMA)
        anio = datetime.today().year
        if mes_num < datetime.today().month:
            anio += 1
        vto = GGALDataFetcher._tercer_viernes(anio, mes_num)

        return flag, strike, vto

    @staticmethod
    def _tercer_viernes(year, month):
        """Calcula el tercer viernes de un mes dado."""
        d = datetime(year, month, 1)
        # Primer viernes
        dias_hasta_viernes = (4 - d.weekday()) % 7
        primer_viernes = d + timedelta(days=dias_hasta_viernes)
        return primer_viernes + timedelta(weeks=2)

    @staticmethod
    def _cadena_sintetica(spot, iv_base=0.65, dias=30):
        """
        Genera cadena sintética con BS cuando no hay datos reales.
        Usada como fallback para demo / testing.
        """
        r = 0.05
        T = dias / 365
        rows = []
        strikes = [round(spot * m, -2) for m in
                   [0.75,0.80,0.85,0.90,0.925,0.95,0.975,
                    1.0,1.025,1.05,1.075,1.10,1.15,1.20,1.25]]

        for k in strikes:
            iv_skew = iv_base + 0.10 * ((spot - k) / spot)  # skew básico
            iv_skew = max(iv_skew, 0.20)
            pc = BS.call(spot, k, T, r, iv_skew)
            pp = BS.put(spot, k, T, r, iv_skew)
            vto = (datetime.today() + timedelta(days=dias)).strftime('%d/%m/%Y')
            for flag, precio in [('call', pc), ('put', pp)]:
                if precio > 0.5:
                    rows.append({
                        'ticker' : f"SINT_{flag[0].upper()}_{k:.0f}",
                        'tipo'   : flag,
                        'strike' : k,
                        'precio' : round(precio, 2),
                        'vol_op' : np.random.randint(50, 500),
                        'oi'     : np.random.randint(100, 2000),
                        'dias'   : dias,
                        'vto'    : vto,
                    })
        return rows


# ============================================================
# MOTOR DE ALERTAS
# ============================================================
class AlertEngine:

    def __init__(self, df: pd.DataFrame, spot: float, r=0.05):
        self.df   = df.copy()
        self.spot = spot
        self.r    = r
        self._calcular_iv()

    # ----------------------------------------------------------
    def _calcular_iv(self):
        rows = []
        for _, row in self.df.iterrows():
            T = row['dias'] / 365
            iv_val = BS.iv(row['precio'], self.spot,
                           row['strike'], T, self.r, row['tipo'])
            rows.append(iv_val)
        self.df['iv'] = rows

        # Percentil de IV por tipo
        for flag in ['call','put']:
            mask = self.df['tipo'] == flag
            vals = self.df.loc[mask, 'iv'].dropna()
            if len(vals) > 1:
                self.df.loc[mask, 'iv_pct'] = vals.rank(pct=True) * 100
            else:
                self.df.loc[mask, 'iv_pct'] = 50.0

    # ----------------------------------------------------------
    def run(self):
        alertas = []
        alertas += self._ratio_invertido()
        alertas += self._ratio_directo()
        alertas += self._bull_call_spread()
        alertas += self._bear_put_spread()
        alertas += self._straddle_strangle()
        alertas += self._iv_outliers()
        # Ordenar por "urgencia" (percentil desc)
        alertas.sort(key=lambda x: x.get('percentil', 0), reverse=True)
        return alertas

    # ── helpers ──────────────────────────────────────────────
    def _pares(self, flag):
        sub = self.df[self.df['tipo']==flag].sort_values('strike')
        return sub

    def _diff_k_pct(self, k1, k2):
        return abs(k2-k1)/k1*100

    # ── RATIO INVERTIDO (2x1) ─────────────────────────────────
    def _ratio_invertido(self):
        alertas = []
        for flag in ['call','put']:
            sub = self._pares(flag)
            strikes = sub['strike'].values
            precios = sub['precio'].values
            ivs     = sub['iv'].values
            iv_pcts = sub['iv_pct'].values if 'iv_pct' in sub.columns else [50]*len(strikes)

            for i in range(len(strikes)-1):
                for j in range(i+1, len(strikes)):
                    k1,k2 = strikes[i], strikes[j]
                    p1,p2 = precios[i], precios[j]
                    dk    = self._diff_k_pct(k1,k2)

                    if dk > 20 or p2 <= 0: continue
                    ratio = p1/p2

                    # p1 ≈ 2*p2 → vender 1 barata, comprar 2 cara
                    if ratio >= 1.85:
                        credito = p1 - 2*p2
                        pct = float(iv_pcts[i]) if len(iv_pcts)>i else 70
                        alertas.append({
                            'tipo'       : 'RATIO_INV',
                            'card_class' : 'card-ratio',
                            'titulo'     : '🎯 NUEVA ENTRADA ASIMÉTRICA (ANOMALÍA)',
                            'estrategia' : 'FRONT RATIO INVERTIDO (2x1)',
                            'bases'      : f'{flag.upper()} {k1:.0f} / {flag.upper()} {k2:.0f}',
                            'senal'      : f'Percentil Desquiciado: {min(pct,99):.1f}% '
                                           f'(Señal CARA → Vendé)',
                            'senal_color': '#f85149',
                            'precio'     : round(abs(credito),2),
                            'spot'       : self.spot,
                            'detalle'    : (
                                f"👉 Venta 1 {flag.upper()} {k1:.0f} a ${p1:.2f} / "
                                f"Compra 2 {flag.upper()} {k2:.0f} a ${p2:.2f} | "
                                f"{'✅ Crédito' if credito>=0 else '⚠️ Débito'}: ${abs(credito):.2f} | "
                                f"Ratio precio: {ratio:.2f}x | "
                                f"Δ strikes: {dk:.1f}%"
                            ),
                            'percentil'  : min(pct,99),
                            'iv_k1'      : round(float(ivs[i])*100,1) if not np.isnan(ivs[i]) else 0,
                            'iv_k2'      : round(float(ivs[j])*100,1) if not np.isnan(ivs[j]) else 0,
                            'timestamp'  : datetime.now().strftime('%H:%M'),
                        })
        return alertas

    # ── RATIO DIRECTO (1x2) ───────────────────────────────────
    def _ratio_directo(self):
        alertas = []
        for flag in ['call','put']:
            sub = self._pares(flag)
            strikes = sub['strike'].values
            precios = sub['precio'].values
            ivs     = sub['iv'].values
            iv_pcts = sub['iv_pct'].values if 'iv_pct' in sub.columns else [50]*len(strikes)

            for i in range(len(strikes)-1):
                for j in range(i+1, len(strikes)):
                    k1,k2 = strikes[i], strikes[j]
                    p1,p2 = precios[i], precios[j]
                    dk    = self._diff_k_pct(k1,k2)

                    if dk > 20 or p2 <= 0: continue
                    ratio = p1/p2

                    if ratio <= 0.55:
                        credito = 2*p2 - p1
                        pct = float(iv_pcts[j]) if len(iv_pcts)>j else 70
                        alertas.append({
                            'tipo'       : 'RATIO_DIR',
                            'card_class' : 'card-entry',
                            'titulo'     : '🎯 NUEVA ENTRADA (RATIO SPREAD)',
                            'estrategia' : 'RATIO SPREAD (1x2)',
                            'bases'      : f'{flag.upper()} {k1:.0f} / {flag.upper()} {k2:.0f}',
                            'senal'      : f'OTM sobre-valuada | Ratio: {ratio:.2f}x | '
                                           f'Percentil: {min(pct,99):.1f}%',
                            'senal_color': '#2ea043',
                            'precio'     : round(abs(credito),2),
                            'spot'       : self.spot,
                            'detalle'    : (
                                f"👉 Compra 1 {flag.upper()} {k1:.0f} a ${p1:.2f} / "
                                f"Venta 2 {flag.upper()} {k2:.0f} a ${p2:.2f} | "
                                f"{'✅ Crédito' if credito>=0 else '⚠️ Débito'}: ${abs(credito):.2f} | "
                                f"Ratio: {ratio:.2f}x | Δ strikes: {dk:.1f}%"
                            ),
                            'percentil'  : (1-ratio)*100,
                            'iv_k1'      : round(float(ivs[i])*100,1) if not np.isnan(ivs[i]) else 0,
                            'iv_k2'      : round(float(ivs[j])*100,1) if not np.isnan(ivs[j]) else 0,
                            'timestamp'  : datetime.now().strftime('%H:%M'),
                        })
        return alertas

    # ── BULL CALL SPREAD ──────────────────────────────────────
    def _bull_call_spread(self):
        alertas = []
        sub = self._pares('call')
        strikes = sub['strike'].values
        precios = sub['precio'].values

        for i in range(len(strikes)-1):
            k1,k2 = strikes[i], strikes[i+1]
            p1,p2 = precios[i], precios[i+1]
            dk    = self._diff_k_pct(k1,k2)
            if dk > 15 or p2 <= 0: continue
            debito      = p1-p2
            max_gan     = (k2-k1)-debito
            if debito <= 0 or max_gan <= 0: continue
            rr = max_gan/debito
            if rr >= 1.2:
                alertas.append({
                    'tipo'       : 'BULL_CALL',
                    'card_class' : 'card-entry',
                    'titulo'     : '📈 BULL CALL SPREAD',
                    'estrategia' : 'LONG BULL CALL SPREAD',
                    'bases'      : f'CALL {k1:.0f} / CALL {k2:.0f}',
                    'senal'      : f'R/R: {rr:.1f}x | Δ strikes: {dk:.1f}%',
                    'senal_color': '#2ea043',
                    'precio'     : round(debito,2),
                    'spot'       : self.spot,
                    'detalle'    : (
                        f"👉 Compra 1 CALL {k1:.0f} a ${p1:.2f} / "
                        f"Venta 1 CALL {k2:.0f} a ${p2:.2f} | "
                        f"Débito: ${debito:.2f} | Máx ganancia: ${max_gan:.2f} | "
                        f"R/R: {rr:.1f}x | BE: ${k1+debito:.0f}"
                    ),
                    'percentil'  : min(rr*30,99),
                    'timestamp'  : datetime.now().strftime('%H:%M'),
                })
        return alertas

    # ── BEAR PUT SPREAD ───────────────────────────────────────
    def _bear_put_spread(self):
        alertas = []
        sub = self._pares('put')
        strikes = sub['strike'].values
        precios = sub['precio'].values

        for i in range(len(strikes)-1):
            k1,k2 = strikes[i], strikes[i+1]   # k2 > k1
            p1,p2 = precios[i], precios[i+1]
            dk    = self._diff_k_pct(k1,k2)
            if dk > 15 or p1 <= 0: continue
            debito  = p2-p1
            max_gan = (k2-k1)-debito
            if debito <= 0 or max_gan <= 0: continue
            rr = max_gan/debito
            if rr >= 1.2:
                alertas.append({
                    'tipo'       : 'BEAR_PUT',
                    'card_class' : 'card-danger',
                    'titulo'     : '📉 BEAR PUT SPREAD',
                    'estrategia' : 'LONG BEAR PUT SPREAD',
                    'bases'      : f'PUT {k1:.0f} / PUT {k2:.0f}',
                    'senal'      : f'R/R: {rr:.1f}x | Δ strikes: {dk:.1f}%',
                    'senal_color': '#f85149',
                    'precio'     : round(debito,2),
                    'spot'       : self.spot,
                    'detalle'    : (
                        f"👉 Compra 1 PUT {k2:.0f} a ${p2:.2f} / "
                        f"Venta 1 PUT {k1:.0f} a ${p1:.2f} | "
                        f"Débito: ${debito:.2f} | Máx ganancia: ${max_gan:.2f} | "
                        f"R/R: {rr:.1f}x | BE: ${k2-debito:.0f}"
                    ),
                    'percentil'  : min(rr*30,99),
                    'timestamp'  : datetime.now().strftime('%H:%M'),
                })
        return alertas

    # ── STRADDLE / STRANGLE ───────────────────────────────────
    def _straddle_strangle(self):
        alertas = []
        calls = self.df[self.df['tipo']=='call']
        puts  = self.df[self.df['tipo']=='put']

        for _, crow in calls.iterrows():
            kc,pc = crow['strike'], crow['precio']
            ivc   = crow['iv'] if not np.isnan(crow.get('iv',np.nan)) else 0

            # ── STRADDLE (mismo strike) ──
            same = puts[puts['strike']==kc]
            if not same.empty:
                prow = same.iloc[0]
                kp,pp = prow['strike'], prow['precio']
                ivp   = prow['iv'] if not np.isnan(prow.get('iv',np.nan)) else 0
                total = pc+pp
                be_pct = total/kc*100
                iv_avg = (ivc+ivp)/2

                tag = 'VENDIDO' if iv_avg>0.55 else 'COMPRADO'
                cc  = 'card-warning' if tag=='VENDIDO' else 'card-entry'
                sc  = '#d29922'      if tag=='VENDIDO' else '#2ea043'
                pct = min(iv_avg*130,99)

                alertas.append({
                    'tipo'       : f'STRADDLE_{tag}',
                    'card_class' : cc,
                    'titulo'     : f'⚖️ STRADDLE {tag} | {kc:.0f}',
                    'estrategia' : f'STRADDLE {tag}',
                    'bases'      : f'CALL {kc:.0f} / PUT {kc:.0f}',
                    'senal'      : f'IV avg: {iv_avg*100:.1f}% | Percentil: {pct:.1f}%',
                    'senal_color': sc,
                    'precio'     : round(total,2),
                    'spot'       : self.spot,
                    'detalle'    : (
                        f"{'Venta' if tag=='VENDIDO' else 'Compra'} 1 CALL {kc:.0f} a ${pc:.2f} + "
                        f"{'Venta' if tag=='VENDIDO' else 'Compra'} 1 PUT {kc:.0f} a ${pp:.2f} | "
                        f"Prima {'cobrada' if tag=='VENDIDO' else 'pagada'}: ${total:.2f} | "
                        f"Rango neutro: ±{be_pct:.1f}% | "
                        f"BE: ${kc-total:.0f} / ${kc+total:.0f}"
                    ),
                    'percentil'  : pct,
                    'timestamp'  : datetime.now().strftime('%H:%M'),
                })

            # ── STRANGLE (strikes distintos OTM) ──
            for _, prow in puts.iterrows():
                kp,pp = prow['strike'], prow['precio']
                ivp   = prow['iv'] if not np.isnan(prow.get('iv',np.nan)) else 0

                if kc <= self.spot or kp >= self.spot: continue
                if kc == kp: continue
                dk_c = (kc-self.spot)/self.spot
                dk_p = (self.spot-kp)/self.spot
                if dk_c > 0.12 or dk_p > 0.12: continue

                total  = pc+pp
                iv_avg = (ivc+ivp)/2
                tag = 'VENDIDO' if iv_avg>0.50 else 'COMPRADO'
                cc  = 'card-warning' if tag=='VENDIDO' else 'card-info'
                sc  = '#d29922'      if tag=='VENDIDO' else '#388bfd'
                pct = min(iv_avg*120,99)

                alertas.append({
                    'tipo'       : f'STRANGLE_{tag}',
                    'card_class' : cc,
                    'titulo'     : f'⚖️ STRANGLE {tag}',
                    'estrategia' : f'STRANGLE {tag}',
                    'bases'      : f'CALL {kc:.0f} / PUT {kp:.0f}',
                    'senal'      : f'IV avg: {iv_avg*100:.1f}% | Percentil: {pct:.1f}%',
                    'senal_color': sc,
                    'precio'     : round(total,2),
                    'spot'       : self.spot,
                    'detalle'    : (
                        f"{'Venta' if tag=='VENDIDO' else 'Compra'} 1 CALL {kc:.0f} a ${pc:.2f} + "
                        f"{'Venta' if tag=='VENDIDO' else 'Compra'} 1 PUT {kp:.0f} a ${pp:.2f} | "
                        f"Prima {'cobrada' if tag=='VENDIDO' else 'pagada'}: ${total:.2f} | "
                        f"Rango: ${kp-total:.0f} / ${kc+total:.0f}"
                    ),
                    'percentil'  : pct,
                    'timestamp'  : datetime.now().strftime('%H:%M'),
                })
        return alertas

    # ── IV OUTLIERS ───────────────────────────────────────────
    def _iv_outliers(self):
        alertas = []
        for _, row in self.df.iterrows():
            iv  = row.get('iv',  np.nan)
            pct = row.get('iv_pct', 50)
            if np.isnan(iv) or iv <= 0: continue

            if pct >= 88:
                alertas.append({
                    'tipo'       : 'IV_ALTA',
                    'card_class' : 'card-warning',
                    'titulo'     : f'🔥 IV EXTREMA — {row["tipo"].upper()} {row["strike"]:.0f}',
                    'estrategia' : 'VENTA DE VOLATILIDAD',
                    'bases'      : f'{row["tipo"].upper()} {row["strike"]:.0f}',
                    'senal'      : f'IV: {iv*100:.1f}% | Percentil: {pct:.1f}%',
                    'senal_color': '#f85149',
                    'precio'     : round(row['precio'],2),
                    'spot'       : self.spot,
                    'detalle'    : (
                        f"👉 Venta 1 {row['tipo'].upper()} {row['strike']:.0f} "
                        f"a ${row['precio']:.2f} | IV extremadamente alta: {iv*100:.1f}% | "
                        f"Percentil: {pct:.1f}% — Prima inflada, oportunidad de venta"
                    ),
                    'percentil'  : pct,
                    'timestamp'  : datetime.now().strftime('%H:%M'),
                })
            elif pct <= 12:
                alertas.append({
                    'tipo'       : 'IV_BAJA',
                    'card_class' : 'card-info',
                    'titulo'     : f'💎 IV MÍNIMA — {row["tipo"].upper()} {row["strike"]:.0f}',
                    'estrategia' : 'COMPRA DE VOLATILIDAD',
                    'bases'      : f'{row["tipo"].upper()} {row["strike"]:.0f}',
                    'senal'      : f'IV: {iv*100:.1f}% | Percentil: {pct:.1f}%',
                    'senal_color': '#388bfd',
                    'precio'     : round(row['precio'],2),
                    'spot'       : self.spot,
                    'detalle'    : (
                        f"👉 Compra 1 {row['tipo'].upper()} {row['strike']:.0f} "
                        f"a ${row['precio']:.2f} | IV históricamente baja: {iv*100:.1f}% | "
                        f"Percentil: {pct:.1f}% — Prima barata, oportunidad de compra"
                    ),
                    'percentil'  : pct,
                    'timestamp'  : datetime.now().strftime('%H:%M'),
                })
        return alertas


# ============================================================
# COMPONENTES UI
# ============================================================
def render_alert_card(a: dict):
    pct   = a.get('percentil', 0)
    bar_w = min(int(pct), 100)
    bar_c = ('#f85149' if pct >= 75 else
             '#d29922' if pct >= 50 else
             '#2ea043' if pct >= 25 else '#388bfd')

    iv_k1 = a.get('iv_k1','—')
    iv_k2 = a.get('iv_k2','—')
    iv_str = (f"<br>📊 IV: <b>{iv_k1}%</b> / <b>{iv_k2}%</b>"
              if iv_k1 != '—' else '')

    html = f"""
    <div class="alert-card {a['card_class']}">
      <div class="alert-title">{a['titulo']}</div>

      👉 <b>Estrategia:</b> {a['estrategia']}<br>
      🏳️ <b>Bases:</b> {a['bases']}<br>
      🎯 <b>Señal:</b>
         <span style="color:{a['senal_color']};font-weight:700">{a['senal']}</span><br>
      💰 <b>Precio armado:</b> ${a['precio']:.2f}<br>
      📈 <b>Spot GGAL:</b> ${a['spot']:,.2f}
      {iv_str}

      <div class="pct-bar" style="width:{bar_w}%;background:{bar_c};"></div>

      <div class="alert-detail">
        🔍 <b>Detalle:</b> {a['detalle']}
      </div>
      <div class="timestamp">{a['timestamp']}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric(label, value, color='#58a6ff'):
    st.markdown(f"""
    <div class="metric-box">
      <div class="metric-value" style="color:{color}">{value}</div>
      <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_cadena_table(df: pd.DataFrame, spot: float):
    if df.empty:
        st.warning("Sin datos de opciones.")
        return

    df = df.copy()
    df['moneyness'] = df.apply(
        lambda r: '🟢 ITM' if (r['tipo']=='call' and r['strike']<spot) or
                               (r['tipo']=='put'  and r['strike']>spot)
                  else ('🔴 OTM' if (r['tipo']=='call' and r['strike']>spot) or
                                    (r['tipo']=='put' and r['strike']<spot)
                        else '🟡 ATM'), axis=1)

    cols_show = ['ticker','tipo','strike','precio','iv','iv_pct','vol_op','oi','vto','moneyness']
    cols_show = [c for c in cols_show if c in df.columns]
    display   = df[cols_show].copy()

    if 'iv' in display.columns:
        display['iv'] = display['iv'].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) and x>0 else "—")
    if 'iv_pct' in display.columns:
        display['iv_pct'] = display['iv_pct'].apply(
            lambda x: f"{x:.0f}%" if pd.notna(x) else "—")

    display.columns = [c.upper() for c in display.columns]
    st.dataframe(display, use_container_width=True, height=420)


def render_iv_smile(df: pd.DataFrame, spot: float):
    if df.empty or 'iv' not in df.columns:
        return
    df = df.dropna(subset=['iv'])
    df = df[df['iv']>0]
    if df.empty: return

    fig = go.Figure()
    for flag, color in [('call','#2ea043'),('put','#f85149')]:
        sub = df[df['tipo']==flag].sort_values('strike')
        if sub.empty: continue
        fig.add_trace(go.Scatter(
            x=sub['strike'], y=sub['iv']*100,
            mode='lines+markers', name=flag.upper(),
            line=dict(color=color, width=2),
            marker=dict(size=7),
            hovertemplate='Strike: %{x}<br>IV: %{y:.1f}%<extra></extra>'
        ))

    fig.add_vline(x=spot, line_dash='dash',
                  line_color='#58a6ff', annotation_text=f'Spot ${spot:,.0f}')
    fig.update_layout(
        title='📊 Curva de Volatilidad Implícita (IV Smile)',
        paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
        font_color='#e6edf3',
        xaxis=dict(title='Strike', gridcolor='#21262d'),
        yaxis=dict(title='IV (%)', gridcolor='#21262d'),
        legend=dict(bgcolor='#161b22', bordercolor='#30363d'),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_payoff(alerta: dict, spot: float):
    """Grafica el payoff de la estrategia seleccionada."""
    strat = alerta['estrategia']
    bases = alerta['bases']
    precio= alerta['precio']

    S_range = np.linspace(spot*0.7, spot*1.3, 300)
    payoff  = np.zeros_like(S_range)

    try:
        parts = bases.replace('CALL','C').replace('PUT','P').split('/')
        ks = [float(p.strip().split()[-1]) for p in parts]

        if 'BULL CALL' in strat:
            k1,k2 = ks[0],ks[1]
            payoff = np.maximum(S_range-k1,0) - np.maximum(S_range-k2,0) - precio

        elif 'BEAR PUT' in strat:
            k1,k2 = ks[0],ks[1]
            payoff = np.maximum(k2-S_range,0) - np.maximum(k1-S_range,0) - precio

        elif 'STRADDLE' in strat:
            k = ks[0]
            sign = -1 if 'VENDIDO' in strat else 1
            payoff = sign*(np.maximum(S_range-k,0)+np.maximum(k-S_range,0)) + \
                     (precio if sign==-1 else -precio)

        elif 'STRANGLE' in strat:
            kc,kp = max(ks),min(ks)
            sign = -1 if 'VENDIDO' in strat else 1
            payoff = sign*(np.maximum(S_range-kc,0)+np.maximum(kp-S_range,0)) + \
                     (precio if sign==-1 else -precio)

        elif 'RATIO INVERTIDO' in strat or 'RATIO_INV' in alerta.get('tipo',''):
            k1,k2 = ks[0],ks[1]
            flag = 'C' if 'CALL' in bases else 'P'
            if flag=='C':
                payoff = (-np.maximum(S_range-k1,0) +
                           2*np.maximum(S_range-k2,0) + precio)
            else:
                payoff = (-np.maximum(k1-S_range,0) +
                           2*np.maximum(k2-S_range,0) + precio)

        elif 'RATIO SPREAD' in strat:
            k1,k2 = ks[0],ks[1]
            flag = 'C' if 'CALL' in bases else 'P'
            if flag=='C':
                payoff = (np.maximum(S_range-k1,0) -
                          2*np.maximum(S_range-k2,0) + precio)
            else:
                payoff = (np.maximum(k1-S_range,0) -
                          2*np.maximum(k2-S_range,0) + precio)
    except:
        pass

    colors = ['#2ea043' if v>=0 else '#f85149' for v in payoff]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=S_range, y=payoff, mode='lines',
        line=dict(color='#58a6ff', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(88,166,255,0.1)',
        name='P&L',
        hovertemplate='Spot: $%{x:,.0f}<br>P&L: $%{y:,.2f}<extra></extra>'
    ))
    fig.add_hline(y=0, line_dash='dash', line_color='#484f58')
    fig.add_vline(x=spot, line_dash='dot',
                  line_color='#d29922', annotation_text='Spot actual')
    fig.update_layout(
        title=f'📉 Payoff al Vencimiento — {strat}',
        paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
        font_color='#e6edf3',
        xaxis=dict(title='Precio GGAL al vto', gridcolor='#21262d'),
        yaxis=dict(title='P&L ($)', gridcolor='#21262d'),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# MAIN APP
# ============================================================
def main():

    # ── Header ──────────────────────────────────────────────
    st.markdown("""
    <h1 style='text-align:center; color:#58a6ff; margin-bottom:0'>
      🎯 GGAL Options Scanner
    </h1>
    <p style='text-align:center; color:#8b949e; margin-top:4px'>
      Alertas en tiempo real · Opciones BYMA · Grupo Financiero Galicia
    </p>
    <hr style='border-color:#21262d; margin:12px 0 20px'>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configuración")

        auto_refresh = st.toggle("🔄 Auto-refresh (60s)", value=True)
        r_libre = st.slider("Tasa libre de riesgo (%)", 0.0, 100.0, 5.0, 0.5) / 100

        st.markdown("---")
        st.markdown("### 🔍 Filtros de Alertas")
        min_pct  = st.slider("Percentil mínimo", 0, 100, 50)
        tipos_sel = st.multiselect(
            "Tipos de estrategia",
            ['RATIO_INV','RATIO_DIR','BULL_CALL','BEAR_PUT',
             'STRADDLE_VENDIDO','STRADDLE_COMPRADO',
             'STRANGLE_VENDIDO','STRANGLE_COMPRADO',
             'IV_ALTA','IV_BAJA'],
            default=['RATIO_INV','RATIO_DIR','BULL_CALL','BEAR_PUT',
                     'STRADDLE_VENDIDO','STRADDLE_COMPRADO',
                     'STRANGLE_VENDIDO','STRANGLE_COMPRADO',
                     'IV_ALTA','IV_BAJA']
        )

        st.markdown("---")
        st.markdown("### 📅 Vencimiento")
        dias_min = st.slider("Días mínimos a vto", 1, 30,  5)
        dias_max = st.slider("Días máximos a vto", 10, 120, 60)

        st.markdown("---")
        if st.button("🔄 Actualizar ahora"):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.caption(f"🕐 Última actualización: {datetime.now().strftime('%H:%M:%S')}")

    # ── Cargar datos ─────────────────────────────────────────
    with st.spinner("📡 Obteniendo datos de mercado..."):
        spot = GGALDataFetcher.get_spot()
        df   = GGALDataFetcher.get_opciones_byma(spot or 6000)

    # ── Spot fallback ────────────────────────────────────────
    if spot is None:
        st.warning("⚠️ No se pudo obtener el precio spot en vivo. Usando precio de referencia.")
        spot = st.number_input("Ingresá el precio spot de GGAL manualmente:",
                               min_value=100.0, max_value=50000.0,
                               value=6400.0, step=50.0)

    # ── Filtrar por días ──────────────────────────────────────
    if not df.empty and 'dias' in df.columns:
        df = df[(df['dias'] >= dias_min) & (df['dias'] <= dias_max)]

    # ── Métricas top ─────────────────────────────────────────
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1: render_metric("💵 Spot GGAL", f"${spot:,.2f}", '#58a6ff')
    with col2: render_metric("📋 Opciones", str(len(df)) if not df.empty else "—", '#d29922')
    with col3:
        iv_med = df['iv'].median()*100 if not df.empty and 'iv' in df.columns else 0
        render_metric("📊 IV Mediana", f"{iv_med:.1f}%" if iv_med else "—", '#bc8cff')
    with col4:
        vol_total = int(df['vol_op'].sum()) if not df.empty and 'vol_op' in df.columns else 0
        render_metric("📦 Volumen", f"{vol_total:,}", '#2ea043')
    with col5:
        oi_total = int(df['oi'].sum()) if not df.empty and 'oi' in df.columns else 0
        render_metric("📌 Open Interest", f"{oi_total:,}", '#f85149')

    st.markdown("<hr style='border-color:#21262d;margin:16px 0'>", unsafe_allow_html=True)

    # ── Tabs principales ─────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Alertas",
        "📊 IV Smile",
        "📋 Cadena de Opciones",
        "📉 Payoff"
    ])

    # ── Motor de alertas ──────────────────────────────────────
    alertas_raw = []
    if not df.empty:
        engine = AlertEngine(df, spot, r=r_libre)
        alertas_raw = engine.run()
    else:
        st.error("❌ Sin datos de opciones. Verificá la conexión.")

    # Filtrar alertas
    alertas = [a for a in alertas_raw
               if a.get('percentil', 0) >= min_pct
               and a.get('tipo','') in tipos_sel]

    # ── TAB 1: ALERTAS ────────────────────────────────────────
    with tab1:
        if not alertas:
            st.info("🔍 Sin alertas para los filtros seleccionados. "
                    "Ajustá el percentil mínimo o los tipos de estrategia.")
        else:
            st.markdown(f"### 🔔 {len(alertas)} alertas activas")

            # Dividir en 2 columnas estilo chat
            col_a, col_b = st.columns(2)
            for idx, alerta in enumerate(alertas):
                with (col_a if idx % 2 == 0 else col_b):
                    render_alert_card(alerta)

    # ── TAB 2: IV SMILE ───────────────────────────────────────
    with tab2:
        if not df.empty and 'iv' in df.columns:
            render_iv_smile(df, spot)

            # Tabla de IV por strike
            st.markdown("#### 📊 IV por Strike")
            iv_table = df[['tipo','strike','iv','iv_pct']].copy()
            iv_table = iv_table.dropna(subset=['iv'])
            iv_table = iv_table[iv_table['iv']>0]
            iv_table['iv']     = iv_table['iv'].apply(lambda x: f"{x*100:.1f}%")
            iv_table['iv_pct'] = iv_table['iv_pct'].apply(
                lambda x: f"{x:.0f}%" if pd.notna(x) else "—")
            iv_table.columns = ['Tipo','Strike','IV','IV Percentil']
            st.dataframe(iv_table.sort_values('Strike'),
                         use_container_width=True, height=320)
        else:
            st.info("Sin datos de IV disponibles.")

    # ── TAB 3: CADENA ─────────────────────────────────────────
    with tab3:
        st.markdown("#### 📋 Cadena de Opciones GGAL")
        if not df.empty:
            tipo_filter = st.radio("Filtrar por:", ['Todas','call','put'], horizontal=True)
            df_show = df if tipo_filter=='Todas' else df[df['tipo']==tipo_filter]
            render_cadena_table(df_show, spot)
        else:
            st.info("Sin datos de cadena disponibles.")

    # ── TAB 4: PAYOFF ─────────────────────────────────────────
    with tab4:
        if not alertas:
            st.info("Generá alertas primero para visualizar payoffs.")
        else:
            opciones_payoff = {
                f"{a['estrategia']} | {a['bases']}": a
                for a in alertas
            }
            sel = st.selectbox("Seleccioná una estrategia:",
                               list(opciones_payoff.keys()))
            if sel:
                alerta_sel = opciones_payoff[sel]
                render_alert_card(alerta_sel)
                render_payoff(alerta_sel, spot)

    # ── Auto-refresh ──────────────────────────────────────────
    if auto_refresh:
        time.sleep(1)
        st.markdown(
            "<script>setTimeout(()=>window.location.reload(),60000)</script>",
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()

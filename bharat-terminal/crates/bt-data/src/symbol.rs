// crates/bt-data/src/symbol.rs
// Author: Sourish Dey

use bt_core::{APP_NAME, AUTHOR, TAGLINE};

/// Company list with 500+ entries: (display_name, ticker, exchange)
/// Covers NSE India, BSE India, NYSE, NASDAQ, US ETFs, Crypto, Indian/US/
/// European/Asia-Pacific indices, European and Asia-Pacific stocks, Commodities
/// and Forex. Entries are append-only: existing rows are never reordered, and
/// new coverage goes in versioned sections at the end.
pub const COMPANY_LIST: &[(&str, &str, &str)] = &[
    // Indian — NSE (Nifty 50 constituents + more)
    ("Reliance Industries", "RELIANCE.NS", "NSE"),
    ("Tata Consultancy Services", "TCS.NS", "NSE"),
    ("Infosys", "INFY.NS", "NSE"),
    ("HDFC Bank", "HDFCBANK.NS", "NSE"),
    ("ICICI Bank", "ICICIBANK.NS", "NSE"),
    ("State Bank of India", "SBIN.NS", "NSE"),
    ("Bharti Airtel", "BHARTIARTL.NS", "NSE"),
    ("ITC", "ITC.NS", "NSE"),
    ("Larsen & Toubro", "LT.NS", "NSE"),
    ("Axis Bank", "AXISBANK.NS", "NSE"),
    ("Kotak Mahindra Bank", "KOTAKBANK.NS", "NSE"),
    ("Maruti Suzuki", "MARUTI.NS", "NSE"),
    ("Asian Paints", "ASIANPAINT.NS", "NSE"),
    ("Bajaj Finance", "BAJFINANCE.NS", "NSE"),
    ("Sun Pharma", "SUNPHARMA.NS", "NSE"),
    ("Wipro", "WIPRO.NS", "NSE"),
    ("HCL Technologies", "HCLTECH.NS", "NSE"),
    ("Titan Company", "TITAN.NS", "NSE"),
    ("UltraTech Cement", "ULTRACEMCO.NS", "NSE"),
    ("Nestle India", "NESTLEIND.NS", "NSE"),
    ("Tata Motors", "TATAMOTORS.NS", "NSE"),
    ("Tata Steel", "TATASTEEL.NS", "NSE"),
    ("Adani Enterprises", "ADANIENT.NS", "NSE"),
    ("Adani Ports", "ADANIPORTS.NS", "NSE"),
    ("Bajaj Auto", "BAJAJ-AUTO.NS", "NSE"),
    ("Coal India", "COALINDIA.NS", "NSE"),
    ("Divi's Laboratories", "DIVISLAB.NS", "NSE"),
    ("Dr Reddy's Laboratories", "DRREDDY.NS", "NSE"),
    ("Eicher Motors", "EICHERMOT.NS", "NSE"),
    ("Grasim Industries", "GRASIM.NS", "NSE"),
    ("Hero MotoCorp", "HEROMOTOCO.NS", "NSE"),
    ("Hindalco", "HINDALCO.NS", "NSE"),
    ("Hindustan Unilever", "HINDUNILVR.NS", "NSE"),
    ("IndusInd Bank", "INDUSINDBK.NS", "NSE"),
    ("JSW Steel", "JSWSTEEL.NS", "NSE"),
    ("Mahindra & Mahindra", "M&M.NS", "NSE"),
    ("NTPC", "NTPC.NS", "NSE"),
    ("ONGC", "ONGC.NS", "NSE"),
    ("Power Grid", "POWERGRID.NS", "NSE"),
    ("Tech Mahindra", "TECHM.NS", "NSE"),
    ("Tata Consumer Products", "TATACONSUM.NS", "NSE"),
    ("UPL", "UPL.NS", "NSE"),
    ("Cipla", "CIPLA.NS", "NSE"),
    ("Britannia Industries", "BRITANNIA.NS", "NSE"),
    ("Hindustan Zinc", "HINDZINC.NS", "NSE"),
    ("Bajaj Finserv", "BAJAJFINSV.NS", "NSE"),
    ("SBI Life Insurance", "SBILIFE.NS", "NSE"),
    ("HDFC Life Insurance", "HDFCLIFE.NS", "NSE"),
    ("ICICI Prudential Life", "ICICIPRULI.NS", "NSE"),
    ("Tata Power", "TATAPOWER.NS", "NSE"),
    ("Adani Green Energy", "ADANIGREEN.NS", "NSE"),
    ("Shree Cement", "SHREECEM.NS", "NSE"),
    ("Pidilite Industries", "PIDILITIND.NS", "NSE"),
    ("Dabur India", "DABUR.NS", "NSE"),
    ("Godrej Consumer Products", "GODREJCP.NS", "NSE"),
    ("Berger Paints", "BERGEPAINT.NS", "NSE"),
    ("Ambuja Cements", "AMBUJACEM.NS", "NSE"),
    ("Bosch", "BOSCHLTD.NS", "NSE"),
    ("Cummins India", "CUMMINSIND.NS", "NSE"),
    ("Siemens", "SIEMENS.NS", "NSE"),
    ("ABB India", "ABB.NS", "NSE"),
    ("Bharat Electronics", "BEL.NS", "NSE"),
    ("Hindustan Aeronautics", "HAL.NS", "NSE"),
    ("Bharat Forge", "BHARATFORG.NS", "NSE"),
    ("Thermax", "THERMAX.NS", "NSE"),
    ("Voltas", "VOLTAS.NS", "NSE"),
    ("Blue Star", "BLUESTARCO.NS", "NSE"),
    ("Crompton Greaves", "CROMPTON.NS", "NSE"),
    ("Havells India", "HAVELLS.NS", "NSE"),
    ("Polycab India", "POLYCAB.NS", "NSE"),
    ("KEI Industries", "KEI.NS", "NSE"),
    ("Kansai Nerolac", "KANSAINER.NS", "NSE"),
    ("Akzo Nobel India", "AKZOINDIA.NS", "NSE"),
    ("Supreme Industries", "SUPREMEIND.NS", "NSE"),
    ("Astral", "ASTRAL.NS", "NSE"),
    ("Prince Pipes", "PRINCEPIPE.NS", "NSE"),
    ("Kajaria Ceramics", "KAJARIACER.NS", "NSE"),
    ("Somany Ceramics", "SOMANYCERA.NS", "NSE"),
    ("Orient Electric", "ORIENTELEC.NS", "NSE"),
    ("V-Guard Industries", "VGUARD.NS", "NSE"),
    ("Indian Oil Corporation", "IOC.NS", "NSE"),
    ("Bharat Petroleum", "BPCL.NS", "NSE"),
    ("Hindustan Petroleum", "HINDPETRO.NS", "NSE"),
    ("GAIL India", "GAIL.NS", "NSE"),
    ("Petronet LNG", "PETRONET.NS", "NSE"),
    ("Indraprastha Gas", "IGL.NS", "NSE"),
    ("Gujarat Gas", "GUJGASTRA.NS", "NSE"),
    ("Mahanagar Gas", "MGL.NS", "NSE"),
    ("Tata Communications", "TATACOMM.NS", "NSE"),
    ("Vodafone Idea", "IDEA.NS", "NSE"),
    ("Indus Towers", "INDUSTOWER.NS", "NSE"),
    ("Info Edge India", "NAUKRI.NS", "NSE"),
    ("Zomato", "ZOMATO.NS", "NSE"),
    ("Paytm", "PAYTM.NS", "NSE"),
    ("Policybazaar", "POLICYBZR.NS", "NSE"),
    ("Nykaa", "NYKAA.NS", "NSE"),
    ("Delhivery", "DELHIVERY.NS", "NSE"),
    ("IRCTC", "IRCTC.NS", "NSE"),
    ("Indian Railway Finance", "IRFC.NS", "NSE"),
    ("REC Limited", "RECLTD.NS", "NSE"),
    ("Power Finance Corporation", "PFC.NS", "NSE"),
    ("Bank of Baroda", "BANKBARODA.NS", "NSE"),
    ("Punjab National Bank", "PNB.NS", "NSE"),
    ("Canara Bank", "CANBK.NS", "NSE"),
    ("Union Bank of India", "UNIONBANK.NS", "NSE"),
    ("Indian Bank", "INDIANB.NS", "NSE"),
    ("Bank of India", "BANKINDIA.NS", "NSE"),
    ("Central Bank of India", "CENTRALBK.NS", "NSE"),
    ("UCO Bank", "UCOBANK.NS", "NSE"),
    ("IDFC First Bank", "IDFCFIRSTB.NS", "NSE"),
    ("RBL Bank", "RBLBANK.NS", "NSE"),
    ("Federal Bank", "FEDERALBNK.NS", "NSE"),
    ("Karnataka Bank", "KTKBANK.NS", "NSE"),
    ("City Union Bank", "CUB.NS", "NSE"),
    ("Karur Vysya Bank", "KVBL.NS", "NSE"),
    ("South Indian Bank", "SOUTHBANK.NS", "NSE"),
    ("Dhanlaxmi Bank", "DHANBANK.NS", "NSE"),
    ("CSB Bank", "CSBBANK.NS", "NSE"),
    ("J&K Bank", "J&KBANK.NS", "NSE"),
    ("Bandhan Bank", "BANDHANBNK.NS", "NSE"),
    ("BHEL", "BHEL.NS", "NSE"),
    ("Can Fin Homes", "CANFINHOME.NS", "NSE"),
    ("Cholamandalam Investment", "CHOLAFIN.NS", "NSE"),
    ("L&T Technology Services", "LTTS.NS", "NSE"),
    ("Mphasis", "MPHASIS.NS", "NSE"),
    ("Coforge", "COFORGE.NS", "NSE"),
    ("Persistent Systems", "PERSISTENT.NS", "NSE"),
    ("PI Industries", "PIIND.NS", "NSE"),
    ("Page Industries", "PAGEIND.NS", "NSE"),
    ("Tata Chemicals", "TATACHEM.NS", "NSE"),
    ("Godrej Properties", "GODREJPROP.NS", "NSE"),
    ("Oberoi Realty", "OBEROIRLTY.NS", "NSE"),
    ("Prestige Estates", "PRESTIGE.NS", "NSE"),
    ("Phoenix Mills", "PHOENIXLTD.NS", "NSE"),
    ("Sobha", "SOBHA.NS", "NSE"),
    ("DLF", "DLF.NS", "NSE"),
    ("Macrotech Developers", "LODHA.NS", "NSE"),
    ("Godrej Industries", "GODREJIND.NS", "NSE"),
    ("3M India", "3MINDIA.NS", "NSE"),
    ("AIA Engineering", "AIAENG.NS", "NSE"),
    ("Ajanta Pharma", "AJANTPHARM.NS", "NSE"),
    ("Alembic Pharmaceuticals", "APLLTD.NS", "NSE"),
    ("Alkem Laboratories", "ALKEM.NS", "NSE"),
    ("Aurobindo Pharma", "AUROPHARMA.NS", "NSE"),
    ("Biocon", "BIOCON.NS", "NSE"),
    ("Cadila Healthcare", "CADILAHC.NS", "NSE"),
    ("Dr Lal Pathlabs", "LALPATHLAB.NS", "NSE"),
    ("Fortis Healthcare", "FORTIS.NS", "NSE"),
    ("Glenmark Pharmaceuticals", "GLENMARK.NS", "NSE"),
    ("Gland Pharma", "GLAND.NS", "NSE"),
    ("Granules India", "GRANULES.NS", "NSE"),
    ("IPCA Laboratories", "IPCALAB.NS", "NSE"),
    ("Jubilant Foodworks", "JUBLFOOD.NS", "NSE"),
    ("Lupin", "LUPIN.NS", "NSE"),
    ("Mankind Pharma", "MANKIND.NS", "NSE"),
    ("Natco Pharma", "NATCOPHARM.NS", "NSE"),
    ("Piramal Enterprises", "PEL.NS", "NSE"),
    ("Sanofi India", "SANOFI.NS", "NSE"),
    ("Strides Pharma Science", "STAR.NS", "NSE"),
    ("Torrent Pharmaceuticals", "TORNTPHARM.NS", "NSE"),
    ("Zydus Lifesciences", "ZYDUSLIFE.NS", "NSE"),
    ("Adani Transmission", "ADANITRANS.NS", "NSE"),
    ("Apollo Hospitals", "APOLLOHOSP.NS", "NSE"),
    ("Ashok Leyland", "ASHOKLEY.NS", "NSE"),
    ("Balkrishna Industries", "BALKRISIND.NS", "NSE"),
    // US — Mega Cap Tech
    ("Apple", "AAPL", "NASDAQ"),
    ("Microsoft", "MSFT", "NASDAQ"),
    ("Google (Alphabet)", "GOOGL", "NASDAQ"),
    ("Amazon", "AMZN", "NASDAQ"),
    ("Tesla", "TSLA", "NASDAQ"),
    ("NVIDIA", "NVDA", "NASDAQ"),
    ("Meta Platforms", "META", "NASDAQ"),
    ("Netflix", "NFLX", "NASDAQ"),
    ("Broadcom", "AVGO", "NASDAQ"),
    ("Adobe", "ADBE", "NASDAQ"),
    ("Salesforce", "CRM", "NYSE"),
    ("Oracle", "ORCL", "NYSE"),
    ("Intel", "INTC", "NASDAQ"),
    ("AMD", "AMD", "NASDAQ"),
    ("Qualcomm", "QCOM", "NASDAQ"),
    ("Texas Instruments", "TXN", "NASDAQ"),
    ("Applied Materials", "AMAT", "NASDAQ"),
    ("Micron Technology", "MU", "NASDAQ"),
    ("Lam Research", "LRCX", "NASDAQ"),
    ("KLA Corporation", "KLAC", "NASDAQ"),
    ("Analog Devices", "ADI", "NASDAQ"),
    ("Microchip Technology", "MCHP", "NASDAQ"),
    ("Skyworks Solutions", "SWKS", "NASDAQ"),
    ("Qorvo", "QRVO", "NASDAQ"),
    ("PayPal", "PYPL", "NASDAQ"),
    ("Intuit", "INTU", "NASDAQ"),
    ("ServiceNow", "NOW", "NYSE"),
    ("Shopify", "SHOP", "NYSE"),
    ("Snowflake", "SNOW", "NYSE"),
    ("Datadog", "DDOG", "NASDAQ"),
    ("CrowdStrike", "CRWD", "NASDAQ"),
    ("Cloudflare", "NET", "NYSE"),
    ("Palantir Technologies", "PLTR", "NYSE"),
    ("Zoom Video Communications", "ZM", "NASDAQ"),
    ("DocuSign", "DOCU", "NASDAQ"),
    ("Twilio", "TWLO", "NYSE"),
    ("HubSpot", "HUBS", "NYSE"),
    ("Atlassian", "TEAM", "NASDAQ"),
    ("Workday", "WDAY", "NASDAQ"),
    ("Autodesk", "ADSK", "NASDAQ"),
    ("Synopsys", "SNPS", "NASDAQ"),
    ("Cadence Design Systems", "CDNS", "NASDAQ"),
    ("Fortinet", "FTNT", "NASDAQ"),
    ("Palo Alto Networks", "PANW", "NASDAQ"),
    ("Cisco Systems", "CSCO", "NASDAQ"),
    ("IBM", "IBM", "NYSE"),
    ("Accenture", "ACN", "NYSE"),
    ("Cognizant Technology Solutions", "CTSH", "NASDAQ"),
    ("Fiserv", "FI", "NASDAQ"),
    ("Global Payments", "GPN", "NYSE"),
    ("Fidelity National Information", "FIS", "NYSE"),
    // US — Blue Chip & Finance
    ("JPMorgan Chase", "JPM", "NYSE"),
    ("Berkshire Hathaway", "BRK-B", "NYSE"),
    ("Visa", "V", "NYSE"),
    ("Johnson & Johnson", "JNJ", "NYSE"),
    ("Walmart", "WMT", "NYSE"),
    ("Exxon Mobil", "XOM", "NYSE"),
    ("Procter & Gamble", "PG", "NYSE"),
    ("Mastercard", "MA", "NYSE"),
    ("UnitedHealth", "UNH", "NYSE"),
    ("Home Depot", "HD", "NYSE"),
    ("Bank of America", "BAC", "NYSE"),
    ("Wells Fargo", "WFC", "NYSE"),
    ("Goldman Sachs", "GS", "NYSE"),
    ("Morgan Stanley", "MS", "NYSE"),
    ("Citigroup", "C", "NYSE"),
    ("Pfizer", "PFE", "NYSE"),
    ("Merck & Co", "MRK", "NYSE"),
    ("AbbVie", "ABBV", "NYSE"),
    ("Eli Lilly", "LLY", "NYSE"),
    ("Bristol-Myers Squibb", "BMY", "NYSE"),
    ("Amgen", "AMGN", "NASDAQ"),
    ("Gilead Sciences", "GILD", "NASDAQ"),
    ("Moderna", "MRNA", "NASDAQ"),
    ("Regeneron Pharmaceuticals", "REGN", "NASDAQ"),
    ("Vertex Pharmaceuticals", "VRTX", "NASDAQ"),
    ("Costco Wholesale", "COST", "NASDAQ"),
    ("McDonald's", "MCD", "NYSE"),
    ("Nike", "NKE", "NYSE"),
    ("Starbucks", "SBUX", "NASDAQ"),
    ("Coca-Cola", "KO", "NYSE"),
    ("PepsiCo", "PEP", "NASDAQ"),
    ("Boeing", "BA", "NYSE"),
    ("Lockheed Martin", "LMT", "NYSE"),
    // Crypto (Coinbase / Yahoo)
    ("Bitcoin", "BTC-USD", "Crypto"),
    ("Ethereum", "ETH-USD", "Crypto"),
    ("Solana", "SOL-USD", "Crypto"),
    ("Cardano", "ADA-USD", "Crypto"),
    ("Dogecoin", "DOGE-USD", "Crypto"),
    ("Ripple (XRP)", "XRP-USD", "Crypto"),
    ("Polkadot", "DOT-USD", "Crypto"),
    ("Litecoin", "LTC-USD", "Crypto"),
    ("Chainlink", "LINK-USD", "Crypto"),
    ("Polygon", "MATIC-USD", "Crypto"),
    ("Avalanche", "AVAX-USD", "Crypto"),
    ("Uniswap", "UNI-USD", "Crypto"),
    ("Cosmos", "ATOM-USD", "Crypto"),
    ("Stellar", "XLM-USD", "Crypto"),
    ("VeChain", "VET-USD", "Crypto"),
    ("Filecoin", "FIL-USD", "Crypto"),
    ("TRON", "TRX-USD", "Crypto"),
    ("EOS", "EOS-USD", "Crypto"),
    ("Monero", "XMR-USD", "Crypto"),
    ("Shiba Inu", "SHIB-USD", "Crypto"),
    ("Aave", "AAVE-USD", "Crypto"),
    ("Compound", "COMP-USD", "Crypto"),
    // Major Indices
    ("Nifty 50", "^NSEI", "Index"),
    ("Sensex", "^BSESN", "Index"),
    ("S&P 500", "^GSPC", "Index"),
    ("NASDAQ Composite", "^IXIC", "Index"),
    ("Dow Jones Industrial Average", "^DJI", "Index"),
    ("Russell 2000", "^RUT", "Index"),
    ("VIX", "^VIX", "Index"),
    ("US 10Y Treasury", "^TNX", "Index"),
    ("US Dollar Index", "DX-Y.NYB", "Index"),
    ("FTSE 100", "^FTSE", "Index"),
    // Commodities
    ("Gold", "GC=F", "Commodity"),
    ("Crude Oil WTI", "CL=F", "Commodity"),
    ("Silver", "SI=F", "Commodity"),
    ("Natural Gas", "NG=F", "Commodity"),
    ("Copper", "HG=F", "Commodity"),
    ("Platinum", "PL=F", "Commodity"),
    ("Palladium", "PA=F", "Commodity"),
    ("Corn", "ZC=F", "Commodity"),
    ("Soybeans", "ZS=F", "Commodity"),
    ("Wheat", "ZW=F", "Commodity"),
    ("Coffee", "KC=F", "Commodity"),
    ("Sugar", "SB=F", "Commodity"),
    // v4.1 — Crypto additions (Coinbase + Binance coverage)
    ("Binance Coin", "BNB-USD", "Crypto"),
    ("Bitcoin Cash", "BCH-USD", "Crypto"),
    ("NEAR Protocol", "NEAR-USD", "Crypto"),
    ("Aptos", "APT-USD", "Crypto"),
    // v4.1 — Nifty Next 50 top names missing above
    ("Bajaj Holdings", "BAJAJHLDNG.NS", "NSE"),
    ("ICICI Lombard", "ICICIGI.NS", "NSE"),
    ("IndiGo", "INDIGO.NS", "NSE"),
    ("Jindal Steel", "JINDALSTEL.NS", "NSE"),
    ("Life Insurance Corp", "LICI.NS", "NSE"),
    ("LTIMindtree", "LTIM.NS", "NSE"),
    // v4.1 — Nifty Midcap 150 top names missing above
    ("Aditya Birla Capital", "ABCAPITAL.NS", "NSE"),
    ("ACC", "ACC.NS", "NSE"),
    ("Bata India", "BATAINDIA.NS", "NSE"),
    ("Container Corp", "CONCOR.NS", "NSE"),
    ("Escorts Kubota", "ESCORTS.NS", "NSE"),
    // v4.1 — Nifty Smallcap 250 top names missing above
    ("Aarti Industries", "AARTIIND.NS", "NSE"),
    ("Atul", "ATUL.NS", "NSE"),
    ("Balrampur Chini", "BALRAMCHIN.NS", "NSE"),
    ("Blue Dart Express", "BLUEDART.NS", "NSE"),
    ("Castrol India", "CASTROLIND.NS", "NSE"),
    ("CEAT", "CEATLTD.NS", "NSE"),
    ("Cyient", "CYIENT.NS", "NSE"),
    ("Deepak Nitrite", "DEEPAKNTR.NS", "NSE"),
    ("Finolex Industries", "FINPIPE.NS", "NSE"),
    // v4.1 — Additional sector coverage
    ("AU Small Finance Bank", "AUBANK.NS", "NSE"),
    ("Yes Bank", "YESBANK.NS", "NSE"),
    ("KPIT Technologies", "KPITTECH.NS", "NSE"),
    ("TVS Motor", "TVSMOTOR.NS", "NSE"),
    ("Samvardhana Motherson", "MOTHERSON.NS", "NSE"),
    ("MRF", "MRF.NS", "NSE"),
    ("Vedanta", "VEDL.NS", "NSE"),
    ("SAIL", "SAIL.NS", "NSE"),
    ("NMDC", "NMDC.NS", "NSE"),
    ("NALCO", "NATIONALUM.NS", "NSE"),
    ("Oil India", "OIL.NS", "NSE"),
    ("Marico", "MARICO.NS", "NSE"),
    ("Colgate-Palmolive India", "COLPAL.NS", "NSE"),
    ("Emami", "EMAMILTD.NS", "NSE"),
    ("Radico Khaitan", "RADICO.NS", "NSE"),
    // v4.1 — BSE twins of the Nifty 50 (same companies, .BO suffix)
    ("Reliance Industries (BSE)", "RELIANCE.BO", "BSE"),
    ("TCS (BSE)", "TCS.BO", "BSE"),
    ("HDFC Bank (BSE)", "HDFCBANK.BO", "BSE"),
    ("ICICI Bank (BSE)", "ICICIBANK.BO", "BSE"),
    ("Infosys (BSE)", "INFY.BO", "BSE"),
    ("Hindustan Unilever (BSE)", "HINDUNILVR.BO", "BSE"),
    ("ITC (BSE)", "ITC.BO", "BSE"),
    ("State Bank of India (BSE)", "SBIN.BO", "BSE"),
    ("Bharti Airtel (BSE)", "BHARTIARTL.BO", "BSE"),
    ("Bajaj Finance (BSE)", "BAJFINANCE.BO", "BSE"),
    ("Kotak Mahindra Bank (BSE)", "KOTAKBANK.BO", "BSE"),
    ("Larsen & Toubro (BSE)", "LT.BO", "BSE"),
    ("HCL Technologies (BSE)", "HCLTECH.BO", "BSE"),
    ("Asian Paints (BSE)", "ASIANPAINT.BO", "BSE"),
    ("Axis Bank (BSE)", "AXISBANK.BO", "BSE"),
    ("Maruti Suzuki (BSE)", "MARUTI.BO", "BSE"),
    ("Sun Pharma (BSE)", "SUNPHARMA.BO", "BSE"),
    ("Titan Company (BSE)", "TITAN.BO", "BSE"),
    ("UltraTech Cement (BSE)", "ULTRACEMCO.BO", "BSE"),
    ("Wipro (BSE)", "WIPRO.BO", "BSE"),
    ("Nestle India (BSE)", "NESTLEIND.BO", "BSE"),
    ("Bajaj Finserv (BSE)", "BAJAJFINSV.BO", "BSE"),
    ("Adani Enterprises (BSE)", "ADANIENT.BO", "BSE"),
    ("Adani Ports (BSE)", "ADANIPORTS.BO", "BSE"),
    ("Tata Motors (BSE)", "TATAMOTORS.BO", "BSE"),
    ("Tata Steel (BSE)", "TATASTEEL.BO", "BSE"),
    ("JSW Steel (BSE)", "JSWSTEEL.BO", "BSE"),
    ("Hindalco (BSE)", "HINDALCO.BO", "BSE"),
    ("ONGC (BSE)", "ONGC.BO", "BSE"),
    ("NTPC (BSE)", "NTPC.BO", "BSE"),
    ("Power Grid (BSE)", "POWERGRID.BO", "BSE"),
    ("Coal India (BSE)", "COALINDIA.BO", "BSE"),
    ("Tech Mahindra (BSE)", "TECHM.BO", "BSE"),
    ("IndusInd Bank (BSE)", "INDUSINDBK.BO", "BSE"),
    ("Dr Reddy's Labs (BSE)", "DRREDDY.BO", "BSE"),
    ("Cipla (BSE)", "CIPLA.BO", "BSE"),
    ("Divi's Laboratories (BSE)", "DIVISLAB.BO", "BSE"),
    ("Eicher Motors (BSE)", "EICHERMOT.BO", "BSE"),
    ("Hero MotoCorp (BSE)", "HEROMOTOCO.BO", "BSE"),
    ("Bajaj Auto (BSE)", "BAJAJ-AUTO.BO", "BSE"),
    ("Britannia Industries (BSE)", "BRITANNIA.BO", "BSE"),
    ("Grasim Industries (BSE)", "GRASIM.BO", "BSE"),
    ("Shree Cement (BSE)", "SHREECEM.BO", "BSE"),
    ("Tata Consumer Products (BSE)", "TATACONSUM.BO", "BSE"),
    ("Apollo Hospitals (BSE)", "APOLLOHOSP.BO", "BSE"),
    ("HDFC Life Insurance (BSE)", "HDFCLIFE.BO", "BSE"),
    ("SBI Life Insurance (BSE)", "SBILIFE.BO", "BSE"),
    ("Mahindra & Mahindra (BSE)", "M&M.BO", "BSE"),
    ("UPL (BSE)", "UPL.BO", "BSE"),
    ("Bharat Petroleum (BSE)", "BPCL.BO", "BSE"),
    // v4.1 — Indian sector indices
    ("Bank Nifty", "^NSEBANK", "Index"),
    ("Nifty IT", "^CNXIT", "Index"),
    ("Nifty Auto", "^CNXAUTO", "Index"),
    ("Nifty Pharma", "^CNXPHARMA", "Index"),
    ("Nifty FMCG", "^CNXFMCG", "Index"),
    ("Nifty Metal", "^CNXMETAL", "Index"),
    ("Nifty Energy", "^CNXENERGY", "Index"),
    ("Nifty Realty", "^CNXREALTY", "Index"),
    ("Nifty Infrastructure", "^CNXINFRA", "Index"),
    ("BSE Midcap", "^BSEMIDCAP", "Index"),
    ("BSE Smallcap", "^BSESMALLCAP", "Index"),
    // v4.1 — US S&P 500 names missing above
    ("Chevron", "CVX", "NYSE"),
    // v4.1 — NASDAQ 100 names missing above
    ("ASML", "ASML", "NASDAQ"),
    ("T-Mobile", "TMUS", "NASDAQ"),
    ("Booking Holdings", "BKNG", "NASDAQ"),
    ("Intuitive Surgical", "ISRG", "NASDAQ"),
    ("Honeywell", "HON", "NASDAQ"),
    ("Mondelez", "MDLZ", "NASDAQ"),
    ("ADP", "ADP", "NASDAQ"),
    // v4.1 — US ETFs
    ("SPDR S&P 500", "SPY", "ETF"),
    ("Invesco QQQ", "QQQ", "ETF"),
    ("SPDR Dow Jones", "DIA", "ETF"),
    ("iShares Russell 2000", "IWM", "ETF"),
    ("Vanguard Total Market", "VTI", "ETF"),
    ("Vanguard S&P 500", "VOO", "ETF"),
    ("Vanguard Developed Markets", "VEA", "ETF"),
    ("Vanguard Emerging Markets", "VWO", "ETF"),
    ("SPDR Gold", "GLD", "ETF"),
    ("iShares Silver", "SLV", "ETF"),
    ("US Oil Fund", "USO", "ETF"),
    ("20-Year Treasury", "TLT", "ETF"),
    ("High Yield Corp Bond", "HYG", "ETF"),
    ("ARK Innovation", "ARKK", "ETF"),
    ("Financials", "XLF", "ETF"),
    ("Technology", "XLK", "ETF"),
    ("Energy", "XLE", "ETF"),
    ("Healthcare", "XLV", "ETF"),
    ("Consumer Discretionary", "XLY", "ETF"),
    ("Consumer Staples", "XLP", "ETF"),
    // v4.1 — US Treasury yields missing above
    ("US 5Y Treasury", "^FVX", "Index"),
    ("US 30Y Treasury", "^TYX", "Index"),
    // v4.1 — Europe stocks
    ("ASML (Euronext)", "ASML.AS", "Euronext Amsterdam"),
    ("SAP", "SAP.DE", "Xetra"),
    ("Siemens", "SIE.DE", "Xetra"),
    ("Allianz", "ALV.DE", "Xetra"),
    ("BASF", "BAS.DE", "Xetra"),
    ("BMW", "BMW.DE", "Xetra"),
    ("Mercedes-Benz", "MBG.DE", "Xetra"),
    ("Volkswagen", "VOW3.DE", "Xetra"),
    ("Nestle (Swiss)", "NESN.SW", "SWX"),
    ("Roche", "ROG.SW", "SWX"),
    ("Novartis", "NOVN.SW", "SWX"),
    ("UBS", "UBSG.SW", "SWX"),
    ("LVMH", "MC.PA", "Euronext Paris"),
    ("L'Oreal", "OR.PA", "Euronext Paris"),
    ("TotalEnergies", "TTE.PA", "Euronext Paris"),
    ("Sanofi", "SAN.PA", "Euronext Paris"),
    ("Airbus", "AIR.PA", "Euronext Paris"),
    ("Shell", "SHEL.L", "LSE"),
    ("AstraZeneca", "AZN.L", "LSE"),
    ("HSBC", "HSBA.L", "LSE"),
    ("BP", "BP.L", "LSE"),
    ("Unilever (UK)", "ULVR.L", "LSE"),
    ("Rio Tinto", "RIO.L", "LSE"),
    ("GSK", "GSK.L", "LSE"),
    ("Barclays", "BARC.L", "LSE"),
    // v4.1 — European indices
    ("DAX (Germany)", "^GDAXI", "Index"),
    ("CAC 40 (France)", "^FCHI", "Index"),
    ("Euro Stoxx 50", "^STOXX50E", "Index"),
    ("SMI (Switzerland)", "^SSMI", "Index"),
    ("IBEX 35 (Spain)", "^IBEX", "Index"),
    ("FTSE MIB (Italy)", "FTSEMIB.MI", "Index"),
    // v4.1 — Asia-Pacific stocks
    ("Toyota", "7203.T", "TSE"),
    ("Sony", "6758.T", "TSE"),
    ("Keyence", "6861.T", "TSE"),
    ("SoftBank", "9984.T", "TSE"),
    ("Mitsubishi UFJ", "8306.T", "TSE"),
    ("Samsung Electronics", "005930.KS", "KRX"),
    ("SK Hynix", "000660.KS", "KRX"),
    ("Hyundai Motor", "005380.KS", "KRX"),
    ("Tencent", "0700.HK", "HKEX"),
    ("Alibaba (HK)", "9988.HK", "HKEX"),
    ("Meituan", "3690.HK", "HKEX"),
    ("Xiaomi", "1810.HK", "HKEX"),
    ("Ping An Insurance", "2318.HK", "HKEX"),
    ("China Construction Bank", "0939.HK", "HKEX"),
    ("BHP Group", "BHP.AX", "ASX"),
    ("Commonwealth Bank", "CBA.AX", "ASX"),
    ("CSL", "CSL.AX", "ASX"),
    ("Westpac", "WBC.AX", "ASX"),
    ("DBS Group", "D05.SI", "SGX"),
    ("OCBC", "O39.SI", "SGX"),
    ("UOB", "U11.SI", "SGX"),
    ("Taiwan Semiconductor (ADR)", "TSM", "NYSE"),
    ("TSMC (TW)", "2330.TW", "TWSE"),
    ("Hon Hai Precision", "2317.TW", "TWSE"),
    ("MediaTek", "2454.TW", "TWSE"),
    // v4.1 — Asia-Pacific indices
    ("Nikkei 225 (Japan)", "^N225", "Index"),
    ("Hang Seng (HK)", "^HSI", "Index"),
    ("Shanghai Composite", "000001.SS", "Index"),
    ("KOSPI (Korea)", "^KS11", "Index"),
    ("Taiwan Weighted", "^TWII", "Index"),
    ("ASX 200 (Australia)", "^AXJO", "Index"),
    ("Straits Times (Singapore)", "^STI", "Index"),
    // v4.1 — Commodities missing above
    ("Brent Crude", "BZ=F", "Commodity"),
    ("Cotton", "CT=F", "Commodity"),
    // v4.1 — Forex (Yahoo FX)
    ("USD/INR", "USDINR=X", "Forex"),
    ("EUR/INR", "EURINR=X", "Forex"),
    ("GBP/INR", "GBPINR=X", "Forex"),
    ("JPY/INR", "JPYINR=X", "Forex"),
    ("EUR/USD", "EURUSD=X", "Forex"),
    ("GBP/USD", "GBPUSD=X", "Forex"),
    ("USD/JPY", "USDJPY=X", "Forex"),
    ("USD/CNY", "USDCNY=X", "Forex"),
    ("AUD/USD", "AUDUSD=X", "Forex"),
    ("USD/CAD", "USDCAD=X", "Forex"),
    ("USD/CHF", "USDCHF=X", "Forex"),
    // v4.1 — Crypto indices
    ("CMC Crypto 200", "^CMC200", "Index"),
];

/// Resolve a friendly name or ticker to a Yahoo Finance symbol.
/// Falls back to returning the input as-is if not found.
pub fn resolve(name_or_ticker: &str) -> String {
    for (name, ticker, _) in COMPANY_LIST {
        if name.eq_ignore_ascii_case(name_or_ticker) || ticker.eq_ignore_ascii_case(name_or_ticker)
        {
            return ticker.to_string();
        }
    }
    name_or_ticker.to_string()
}

/// Get display name for a ticker (reverse lookup).
pub fn display_name(ticker: &str) -> String {
    for (name, t, _) in COMPANY_LIST {
        if t.eq_ignore_ascii_case(ticker) {
            return name.to_string();
        }
    }
    ticker.to_string()
}

/// Get exchange for a ticker.
pub fn exchange(ticker: &str) -> &str {
    for (_, t, ex) in COMPANY_LIST {
        if t.eq_ignore_ascii_case(ticker) {
            return ex;
        }
    }
    "Unknown"
}

/// Get all unique exchanges.
pub fn all_exchanges() -> Vec<&'static str> {
    let mut exchanges: Vec<&'static str> = COMPANY_LIST.iter().map(|(_, _, ex)| *ex).collect();
    exchanges.sort();
    exchanges.dedup();
    exchanges
}

/// Get companies filtered by exchange.
pub fn companies_by_exchange(exchange: &str) -> Vec<(&'static str, &'static str, &'static str)> {
    COMPANY_LIST
        .iter()
        .filter(|(_, _, ex)| ex.eq_ignore_ascii_case(exchange))
        .copied()
        .collect()
}

/// Search companies by name or ticker (case-insensitive substring).
pub fn search_companies(query: &str) -> Vec<(&'static str, &'static str, &'static str)> {
    let q = query.to_lowercase();
    COMPANY_LIST
        .iter()
        .filter(|(name, ticker, _)| {
            name.to_lowercase().contains(&q) || ticker.to_lowercase().contains(&q)
        })
        .copied()
        .collect()
}

/// Default company for initial selection.
pub const DEFAULT_COMPANY: (&str, &str, &str) = ("Reliance Industries", "RELIANCE.NS", "NSE");

/// Application metadata constants.
pub fn app_info() -> String {
    format!(
        "{} v{} — Made by {}. {}",
        APP_NAME,
        env!("CARGO_PKG_VERSION"),
        AUTHOR,
        TAGLINE
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_known_names() {
        assert_eq!(resolve("Reliance Industries"), "RELIANCE.NS");
        assert_eq!(resolve("Apple"), "AAPL");
        assert_eq!(resolve("Bitcoin"), "BTC-USD");
        assert_eq!(resolve("Nifty 50"), "^NSEI");
    }

    #[test]
    fn test_resolve_tickers() {
        assert_eq!(resolve("RELIANCE.NS"), "RELIANCE.NS");
        assert_eq!(resolve("AAPL"), "AAPL");
        assert_eq!(resolve("BTC-USD"), "BTC-USD");
    }

    #[test]
    fn test_resolve_unknown_fallback() {
        assert_eq!(resolve("UNKNOWN"), "UNKNOWN");
        assert_eq!(resolve("RANDOM.TICKER"), "RANDOM.TICKER");
    }

    #[test]
    fn test_display_name() {
        assert_eq!(display_name("RELIANCE.NS"), "Reliance Industries");
        assert_eq!(display_name("AAPL"), "Apple");
        assert_eq!(display_name("BTC-USD"), "Bitcoin");
    }

    #[test]
    fn test_exchange_lookup() {
        assert_eq!(exchange("RELIANCE.NS"), "NSE");
        assert_eq!(exchange("AAPL"), "NASDAQ");
        assert_eq!(exchange("BTC-USD"), "Crypto");
    }

    #[test]
    fn test_search() {
        let results = search_companies("reliance");
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "Reliance Industries");

        let results = search_companies("bank");
        assert!(results.len() >= 4); // HDFC, ICICI, Axis, Kotak, SBI, IndusInd
    }

    #[test]
    fn test_companies_by_exchange() {
        let nse = companies_by_exchange("NSE");
        assert!(nse.len() >= 40);

        let nasdaq = companies_by_exchange("NASDAQ");
        assert!(nasdaq.len() >= 8);

        let crypto = companies_by_exchange("Crypto");
        assert!(crypto.len() >= 7);
    }

    #[test]
    fn test_default_company() {
        assert_eq!(DEFAULT_COMPANY.1, "RELIANCE.NS");
        assert!(resolve(&DEFAULT_COMPANY.1) == "RELIANCE.NS");
    }

    #[test]
    fn test_app_info_contains_author() {
        let info = app_info();
        assert!(info.contains("Sourish Dey"));
        assert!(info.contains("Bloomberg power"));
        assert!(info.contains("Made in India"));
    }

    /// v4.1 universe: every ticker must be unique (case-insensitive), or
    /// `resolve`/`display_name` silently return the first duplicate.
    #[test]
    fn test_no_duplicate_tickers() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for (_, ticker, _) in COMPANY_LIST {
            let key = ticker.to_uppercase();
            assert!(seen.insert(key.clone()), "duplicate ticker: {key}");
        }
    }

    /// v4.1 universe: spot-check the newly added coverage resolves.
    #[test]
    fn test_v41_universe_resolves() {
        // Crypto additions.
        assert_eq!(resolve("Binance Coin"), "BNB-USD");
        assert_eq!(resolve("APT-USD"), "APT-USD");
        // New NSE names.
        assert_eq!(resolve("IndiGo"), "INDIGO.NS");
        assert_eq!(resolve("Blue Dart Express"), "BLUEDART.NS");
        // BSE twins.
        assert_eq!(resolve("RELIANCE.BO"), "RELIANCE.BO");
        assert_eq!(display_name("TCS.BO"), "TCS (BSE)");
        // ETFs, regions, FX.
        assert_eq!(resolve("SPY"), "SPY");
        assert_eq!(exchange("SPY"), "ETF");
        assert_eq!(resolve("Toyota"), "7203.T");
        assert_eq!(exchange("7203.T"), "TSE");
        assert_eq!(resolve("USD/INR"), "USDINR=X");
        assert_eq!(exchange("USDINR=X"), "Forex");
        assert_eq!(resolve("Brent Crude"), "BZ=F");
        assert_eq!(resolve("CMC Crypto 200"), "^CMC200");
        // Existing entries still resolve first (append-only order).
        assert_eq!(resolve("Reliance Industries"), "RELIANCE.NS");
        assert!(COMPANY_LIST.len() >= 500, "universe shrank: {}", COMPANY_LIST.len());
    }

    /// v4.1 universe: every entry has a non-empty name, ticker and exchange.
    #[test]
    fn test_entries_are_well_formed() {
        for (name, ticker, exchange) in COMPANY_LIST {
            assert!(!name.is_empty(), "empty name for {ticker}");
            assert!(!ticker.is_empty(), "empty ticker for {name}");
            assert!(!exchange.is_empty(), "empty exchange for {ticker}");
            assert!(!ticker.contains(' '), "ticker with space: {ticker}");
        }
    }
}

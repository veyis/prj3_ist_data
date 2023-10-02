import streamlit as st

st.set_page_config(page_title='TradingView Widget', layout='wide')
st.title('TradingView Widget Example')

# Taking user input to change the symbol dynamically
symbol = st.text_input("Enter Trading Symbol:", value="BIST:TUPRS")

html_code = f'''
<div id="tradingview_34f5a" class="tradingview-widget-container">
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
        var userLang = navigator.language || navigator.userLanguage;
        var tradingViewLocale = userLang.substring(0, 2);
        
        new TradingView.widget({{
            "width": 980,
            "height": 610,
            "symbol": "{symbol}",
            "interval": "D",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": tradingViewLocale,
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": false,
            "allow_symbol_change": true,
            "container_id": "tradingview_34f5a"
        }});
    </script>
</div>
'''

st.components.v1.html(html_code, width=1000, height=630)

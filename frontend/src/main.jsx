import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import 'mdui/mdui.css';
import { setColorScheme } from 'mdui/functions/setColorScheme.js';

setColorScheme('#0022ff', {
  target: document.querySelector('.foo')
});

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
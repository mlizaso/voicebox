import ReactDOM from 'react-dom/client';
import { AppRoot } from '../../app/src/AppRoot';
import '../../app/src/index.css';
import { webPlatform } from './platform';

ReactDOM.createRoot(document.getElementById('root')!).render(<AppRoot platform={webPlatform} />);

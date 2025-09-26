(()=>{var e={};e.id=110,e.ids=[110],e.modules={2934:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external.js")},4580:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external.js")},5869:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external.js")},399:e=>{"use strict";e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},1233:(e,s,t)=>{"use strict";t.r(s),t.d(s,{GlobalError:()=>o.a,__next_app__:()=>p,originalPathname:()=>m,pages:()=>c,routeModule:()=>x,tree:()=>d}),t(5109),t(1506),t(5866);var r=t(3191),a=t(8716),l=t(7922),o=t.n(l),n=t(5231),i={};for(let e in n)0>["default","tree","pages","GlobalError","originalPathname","__next_app__","routeModule"].indexOf(e)&&(i[e]=()=>n[e]);t.d(s,i);let d=["",{children:["api",{children:["__PAGE__",{},{page:[()=>Promise.resolve().then(t.bind(t,5109)),"/Users/robertgilks/Source/sloposcope/web/app/api/page.tsx"]}]},{}]},{layout:[()=>Promise.resolve().then(t.bind(t,1506)),"/Users/robertgilks/Source/sloposcope/web/app/layout.tsx"],"not-found":[()=>Promise.resolve().then(t.t.bind(t,5866,23)),"next/dist/client/components/not-found-error"]}],c=["/Users/robertgilks/Source/sloposcope/web/app/api/page.tsx"],m="/api/page",p={require:t,loadChunk:()=>Promise.resolve()},x=new r.AppPageRouteModule({definition:{kind:a.x.APP_PAGE,page:"/api/page",pathname:"/api",bundlePath:"",filename:"",appPaths:[]},userland:{loaderTree:d}})},2861:(e,s,t)=>{Promise.resolve().then(t.t.bind(t,2994,23)),Promise.resolve().then(t.t.bind(t,6114,23)),Promise.resolve().then(t.t.bind(t,9727,23)),Promise.resolve().then(t.t.bind(t,9671,23)),Promise.resolve().then(t.t.bind(t,1868,23)),Promise.resolve().then(t.t.bind(t,4759,23))},2886:()=>{},5303:()=>{},5109:(e,s,t)=>{"use strict";t.r(s),t.d(s,{default:()=>a});var r=t(9510);function a(){return r.jsx("div",{className:"min-h-screen bg-gray-50",children:r.jsx("div",{className:"container mx-auto px-4 py-8",children:(0,r.jsxs)("div",{className:"max-w-4xl mx-auto",children:[r.jsx("h1",{className:"text-4xl font-bold text-gray-900 mb-8",children:"API Documentation"}),(0,r.jsxs)("div",{className:"bg-white rounded-lg shadow-sm border border-gray-200 p-8",children:[r.jsx("h2",{className:"text-2xl font-semibold mb-4",children:"Sloposcope API"}),r.jsx("p",{className:"text-gray-700 mb-6",children:"The Sloposcope API provides programmatic access to text analysis capabilities. All endpoints are hosted on Cloudflare Workers for fast, global performance."}),(0,r.jsxs)("div",{className:"bg-gray-50 rounded-lg p-4 mb-6",children:[r.jsx("h3",{className:"text-lg font-semibold mb-2",children:"Base URL"}),r.jsx("code",{className:"text-sm bg-white px-2 py-1 rounded border",children:"https://sloposcope-prod.rob-gilks.workers.dev"})]}),r.jsx("h2",{className:"text-2xl font-semibold mb-4",children:"Endpoints"}),(0,r.jsxs)("div",{className:"space-y-6",children:[(0,r.jsxs)("div",{className:"border border-gray-200 rounded-lg p-6",children:[(0,r.jsxs)("div",{className:"flex items-center mb-3",children:[r.jsx("span",{className:"bg-green-100 text-green-800 text-xs font-medium px-2.5 py-0.5 rounded mr-3",children:"GET"}),r.jsx("code",{className:"text-lg font-mono",children:"/health"})]}),r.jsx("p",{className:"text-gray-700 mb-3",children:"Check the health status of the API service."}),r.jsx("h4",{className:"font-semibold mb-2",children:"Response"}),r.jsx("pre",{className:"bg-gray-100 p-3 rounded text-sm overflow-x-auto",children:`{
  "status": "healthy",
  "service": "sloposcope",
  "version": "1.0.0",
  "implementation": "javascript-mock"
}`})]}),(0,r.jsxs)("div",{className:"border border-gray-200 rounded-lg p-6",children:[(0,r.jsxs)("div",{className:"flex items-center mb-3",children:[r.jsx("span",{className:"bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded mr-3",children:"POST"}),r.jsx("code",{className:"text-lg font-mono",children:"/analyze"})]}),r.jsx("p",{className:"text-gray-700 mb-3",children:"Analyze text for AI slop patterns and return detailed metrics."}),r.jsx("h4",{className:"font-semibold mb-2",children:"Request Body"}),r.jsx("pre",{className:"bg-gray-100 p-3 rounded text-sm overflow-x-auto",children:`{
  "text": "Text to analyze",
  "domain": "general", // optional: "general", "news", "qa"
  "language": "en", // optional: "en", "es", "fr", "de"
  "explain": true, // optional: include explanations
  "spans": true // optional: include character spans
}`}),r.jsx("h4",{className:"font-semibold mb-2 mt-4",children:"Response"}),r.jsx("pre",{className:"bg-gray-100 p-3 rounded text-sm overflow-x-auto",children:`{
  "version": "1.0",
  "domain": "general",
  "slop_score": 0.482,
  "confidence": 0.8,
  "level": "Watch",
  "metrics": {
    "density": {"value": 0.4},
    "repetition": {"value": 0.3},
    "templated": {"value": 0.2},
    "coherence": {"value": 0.5},
    "verbosity": {"value": 0.6},
    "tone": {"value": 0.4},
    "subjectivity": {"value": 0.3},
    "fluency": {"value": 0.7},
    "factuality": {"value": 0.8},
    "complexity": {"value": 0.5},
    "relevance": {"value": 0.6}
  },
  "timings_ms": {"total": 150, "nlp": 50, "features": 100},
  "explanations": {...}, // if explain=true
  "spans": [...] // if spans=true
}`})]}),(0,r.jsxs)("div",{className:"border border-gray-200 rounded-lg p-6",children:[(0,r.jsxs)("div",{className:"flex items-center mb-3",children:[r.jsx("span",{className:"bg-green-100 text-green-800 text-xs font-medium px-2.5 py-0.5 rounded mr-3",children:"GET"}),r.jsx("code",{className:"text-lg font-mono",children:"/metrics"})]}),r.jsx("p",{className:"text-gray-700 mb-3",children:"Get information about available analysis metrics."}),r.jsx("h4",{className:"font-semibold mb-2",children:"Response"}),r.jsx("pre",{className:"bg-gray-100 p-3 rounded text-sm overflow-x-auto",children:`{
  "available_metrics": [
    {
      "name": "density",
      "description": "Information density and perplexity measures",
      "range": [0, 1],
      "lower_is_better": false
    },
    // ... other metrics
  ],
  "domains": ["general", "news", "qa"],
  "slop_levels": {
    "Clean": "≤ 0.30",
    "Watch": "0.30 - 0.55",
    "Sloppy": "0.55 - 0.75",
    "High-Slop": "> 0.75"
  }
}`})]})]}),r.jsx("h2",{className:"text-2xl font-semibold mb-4 mt-8",children:"Usage Examples"}),(0,r.jsxs)("div",{className:"space-y-4",children:[(0,r.jsxs)("div",{children:[r.jsx("h4",{className:"font-semibold mb-2",children:"cURL"}),r.jsx("pre",{className:"bg-gray-100 p-3 rounded text-sm overflow-x-auto",children:`curl -X POST https://sloposcope-prod.rob-gilks.workers.dev/analyze \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "Your text here",
    "domain": "general",
    "explain": true
  }'`})]}),(0,r.jsxs)("div",{children:[r.jsx("h4",{className:"font-semibold mb-2",children:"JavaScript"}),r.jsx("pre",{className:"bg-gray-100 p-3 rounded text-sm overflow-x-auto",children:`const response = await fetch('https://sloposcope-prod.rob-gilks.workers.dev/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'Your text here',
    domain: 'general',
    explain: true
  })
});

const result = await response.json();
console.log('Slop Score:', result.slop_score);`})]}),(0,r.jsxs)("div",{children:[r.jsx("h4",{className:"font-semibold mb-2",children:"Python"}),r.jsx("pre",{className:"bg-gray-100 p-3 rounded text-sm overflow-x-auto",children:`import requests

response = requests.post(
    'https://sloposcope-prod.rob-gilks.workers.dev/analyze',
    json={
        'text': 'Your text here',
        'domain': 'general',
        'explain': True
    }
)

result = response.json()
print(f"Slop Score: {result['slop_score']}")`})]})]}),r.jsx("h2",{className:"text-2xl font-semibold mb-4 mt-8",children:"Rate Limits"}),r.jsx("p",{className:"text-gray-700 mb-4",children:"Currently, there are no strict rate limits, but please use the API responsibly. For high-volume usage, consider implementing your own caching and rate limiting."}),r.jsx("h2",{className:"text-2xl font-semibold mb-4",children:"Error Handling"}),r.jsx("p",{className:"text-gray-700 mb-4",children:"The API returns standard HTTP status codes:"}),(0,r.jsxs)("ul",{className:"list-disc list-inside text-gray-700 space-y-1",children:[(0,r.jsxs)("li",{children:[r.jsx("strong",{children:"200:"})," Success"]}),(0,r.jsxs)("li",{children:[r.jsx("strong",{children:"400:"})," Bad Request (invalid input)"]}),(0,r.jsxs)("li",{children:[r.jsx("strong",{children:"404:"})," Not Found"]}),(0,r.jsxs)("li",{children:[r.jsx("strong",{children:"500:"})," Internal Server Error"]})]}),(0,r.jsxs)("div",{className:"bg-yellow-50 border border-yellow-200 rounded-lg p-4 mt-6",children:[r.jsx("h3",{className:"text-lg font-semibold text-yellow-900 mb-2",children:"Note"}),r.jsx("p",{className:"text-yellow-800",children:"This API currently uses a mock implementation for demonstration purposes. The actual sloposcope analysis engine will be integrated in future updates."})]})]})]})})})}},1506:(e,s,t)=>{"use strict";t.r(s),t.d(s,{default:()=>n,metadata:()=>o});var r=t(9510),a=t(5384),l=t.n(a);t(7272);let o={title:"Sloposcope - AI Text Analysis",description:"Detect AI-generated text patterns and measure slop across multiple dimensions"};function n({children:e}){return r.jsx("html",{lang:"en",children:r.jsx("body",{className:l().className,children:e})})}},7272:()=>{}};var s=require("../../webpack-runtime.js");s.C(e);var t=e=>s(s.s=e),r=s.X(0,[734],()=>t(1233));module.exports=r})();
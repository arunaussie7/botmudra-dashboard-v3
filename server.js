const express = require('express');
const cors = require('cors');
const { Server } = require('socket.io');
const http = require('http');
const MetaApi = require('metaapi.cloud-sdk').default;
require('dotenv').config();

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "http://localhost:3000",
    methods: ["GET", "POST"]
  }
});

app.use(cors());
app.use(express.json());

// MT5 credentials
const MT5_CONFIG = {
  server: 'Exness-MT5Trial7',
  login: '203405414',
  password: 'Arun@123'
};

// Initialize MetaAPI
const token = process.env.META_API_TOKEN || 'eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2YzI5MjZhOTliZDY0NzQ4NjRhZjQ3YmYiLCJwZXJtaXNzaW9ucyI6W10sInRva2VuIjoiZXlKaGJHY2lPaUpJVXpVeE1pSXNJblI1Y0NJNklrcFhWQ0o5LmV5SmZhV1FpT2lJMll6STVNalpoT1RsaVpEWTBOelE0TmpSaFpqUTNZbVlpTENKd1pYSnRhWE56YVc5dWN5STZXeUowY21Ga1pTSmRMQ0owYjJ0bGJpSTZJakV5TXpRaUxDSnBZWFFpT2pFMk9UYzJOVGt4TlRkOS5NbGRBOFVwWHBfY0lmTWx1QnBzRHVNdGJGTWFjSVBKSHVfTmRJSXhxWU1NIiwiaWF0IjoxNjk3NjU5MTU3LCJleHAiOjE3MDUyMTkxNTd9.RKA9XJQZ9_9Y9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q9Q';
const api = new MetaApi(token);

let mt5Connected = false;
let accountData = null;
let connection = null;

// Function to initialize MT5 connection
async function initializeMT5() {
  try {
    // Create MT5 account
    const account = await api.metatraderAccountApi.createAccount({
      name: 'Trading Account',
      type: 'cloud',
      login: MT5_CONFIG.login,
      password: MT5_CONFIG.password,
      server: MT5_CONFIG.server,
      platform: 'mt5',
      magic: 123456
    });

    // Wait for the account to be deployed and connected to broker
    console.log('Deploying account...');
    await account.deploy();
    console.log('Waiting for API server to connect to broker...');
    await account.waitConnected();

    // Connect to MetaApi API
    connection = account.getRPCConnection();
    await connection.connect();
    console.log('Connected to MetaApi API');
    mt5Connected = true;

    return connection;
  } catch (error) {
    console.error('Error initializing MT5:', error);
    return null;
  }
}

// Function to fetch MT5 account data
async function fetchMT5Data() {
  try {
    if (!connection) {
      return null;
    }

    const accountInfo = await connection.getAccountInformation();
    const positions = await connection.getPositions();
    const orders = await connection.getOrders();
    const history = await connection.getDeals();

    accountData = {
      balance: accountInfo.balance,
      equity: accountInfo.equity,
      openOrders: positions.map(pos => ({
        symbol: pos.symbol,
        type: pos.type,
        volume: pos.volume,
        openPrice: pos.openPrice,
        profit: pos.profit
      })),
      closedOrders: history.map(deal => ({
        symbol: deal.symbol,
        type: deal.type,
        volume: deal.volume,
        price: deal.price,
        profit: deal.profit,
        time: deal.time
      })),
      lastUpdate: new Date().toISOString()
    };

    return accountData;
  } catch (error) {
    console.error('Error fetching MT5 data:', error);
    return null;
  }
}

// Initialize MT5 connection when server starts
initializeMT5().then(() => {
  console.log('MT5 initialization completed');
});

// Socket.IO connection handling
io.on('connection', (socket) => {
  console.log('Client connected');

  // Send initial data
  if (accountData) {
    socket.emit('mt5Data', accountData);
  }

  // Update data every 1 second
  const updateInterval = setInterval(async () => {
    const data = await fetchMT5Data();
    if (data) {
      socket.emit('mt5Data', data);
    }
  }, 1000);

  socket.on('disconnect', () => {
    console.log('Client disconnected');
    clearInterval(updateInterval);
  });
});

const PORT = process.env.PORT || 5000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
}); 
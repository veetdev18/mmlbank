package main

import (
	"crypto/hmac"
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/joho/godotenv"
)

// LbankClient represents the Lbank API client
type LbankClient struct {
	APIKey    string
	SecretKey string
	BaseURL   string
}

// OrderParams contains parameters needed to place an order
type OrderParams struct {
	Symbol    string  // Trading pair, e.g., "eth_btc"
	Type      string  // "buy" or "sell"
	Price     float64 // Order price
	Amount    float64 // Order quantity
	ClientOID string  // Optional client order ID
}

// OrderResponse represents the response from the create order API
type OrderResponse struct {
	Result    bool   `json:"result"`
	OrderID   string `json:"order_id,omitempty"`
	ErrorCode int    `json:"error_code,omitempty"`
	Timestamp int64  `json:"ts,omitempty"`
}

// NewLbankClient creates a new Lbank API client
func NewLbankClient(apiKey, secretKey string) *LbankClient {
	return &LbankClient{
		APIKey:    apiKey,
		SecretKey: secretKey,
		BaseURL:   "https://api.lbkex.com",
	}
}

// PlaceOrder creates a new order on Lbank exchange
func (c *LbankClient) PlaceOrder(params OrderParams) (*OrderResponse, error) {
	// Construct the parameters map
	paramMap := map[string]string{
		"symbol": params.Symbol,
		"type":   params.Type,
		"price":  strconv.FormatFloat(params.Price, 'f', -1, 64),
		"amount": strconv.FormatFloat(params.Amount, 'f', -1, 64),
	}

	// Add client order ID if provided
	if params.ClientOID != "" {
		paramMap["custom_id"] = params.ClientOID
	}

	// Add API key and timestamp
	paramMap["api_key"] = c.APIKey
	paramMap["timestamp"] = strconv.FormatInt(time.Now().UnixMilli(), 10)

	// Generate signature
	signature := c.generateSignature(paramMap)
	paramMap["sign"] = signature

	// Make the API request
	resp, err := c.makeRequest("/v2/create_order.do", paramMap)
	if err != nil {
		return nil, err
	}

	// Parse the response
	var orderResp OrderResponse
	if err := json.Unmarshal(resp, &orderResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &orderResp, nil
}

// generateSignature creates a signature for the API request
func (c *LbankClient) generateSignature(params map[string]string) string {
	// Sort parameters alphabetically
	var keys []string
	for k := range params {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	// Create parameter string
	var paramStr strings.Builder
	for _, k := range keys {
		paramStr.WriteString(k)
		paramStr.WriteString("=")
		paramStr.WriteString(params[k])
		paramStr.WriteString("&")
	}

	// Remove trailing ampersand
	paramString := paramStr.String()
	if len(paramString) > 0 {
		paramString = paramString[:len(paramString)-1]
	}

	// Create MD5 HMAC signature
	h := hmac.New(md5.New, []byte(c.SecretKey))
	h.Write([]byte(paramString))
	return hex.EncodeToString(h.Sum(nil))
}

// makeRequest sends a POST request to the Lbank API
func (c *LbankClient) makeRequest(endpoint string, params map[string]string) ([]byte, error) {
	// Create form values
	form := url.Values{}
	for k, v := range params {
		form.Add(k, v)
	}

	// Create request
	req, err := http.NewRequest("POST", c.BaseURL+endpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, err
	}

	// Set headers
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	// Send request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// Read response
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	return body, nil
}

// loadEnv loads environment variables from .env file
func loadEnv() {
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}
}

func main() {
	// Load environment variables from .env file
	loadEnv()

	// Get API credentials from environment variables
	apiKey := os.Getenv("LBANK_API_KEY")
	secretKey := os.Getenv("LBANK_SECRET_KEY")

	if apiKey == "" || secretKey == "" {
		log.Fatal("API credentials not found in environment variables")
	}

	// Example usage with environment variables
	client := NewLbankClient(apiKey, secretKey)
	orderParams := OrderParams{
		Symbol: "eth_usdt",
		Type:   "buy",
		Price:  2000.50,
		Amount: 0.5,
	}
	resp, err := client.PlaceOrder(orderParams)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Printf("Order placed successfully, ID: %s\n", resp.OrderID)
}

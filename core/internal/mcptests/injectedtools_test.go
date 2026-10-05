package mcptests

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	bifrost "github.com/maximhq/bifrost/core"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const injectedSearchTool = "bifrostInternal-web_search"

// fakeOpenAI answers /v1/chat/completions from a script and records every body it got.
type fakeOpenAI struct {
	mu      sync.Mutex
	bodies  []map[string]any
	replies []string
}

func (f *fakeOpenAI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)
	var body map[string]any
	_ = json.Unmarshal(raw, &body)
	f.mu.Lock()
	f.bodies = append(f.bodies, body)
	reply := f.replies[0]
	f.replies = f.replies[1:]
	f.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(reply))
}

func (f *fakeOpenAI) toolNames(turn int) []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	var names []string
	tools, _ := f.bodies[turn]["tools"].([]any)
	for _, tool := range tools {
		entry, _ := tool.(map[string]any)
		if fn, ok := entry["function"].(map[string]any); ok {
			names = append(names, fn["name"].(string))
		} else {
			names = append(names, entry["type"].(string))
		}
	}
	return names
}

func chatCompletionJSON(content string, finish string, toolCall string) string {
	message := map[string]any{"role": "assistant", "content": content}
	if toolCall != "" {
		message["content"] = nil
		message["tool_calls"] = []any{map[string]any{
			"id": "call_1", "type": "function",
			"function": map[string]any{"name": toolCall, "arguments": `{"query":"weather in paris"}`},
		}}
	}
	out, _ := json.Marshal(map[string]any{
		"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "gpt-4o",
		"choices": []any{map[string]any{"index": 0, "message": message, "finish_reason": finish}},
		"usage":   map[string]any{"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
	})
	return string(out)
}

// injectedToolsAccount serves one OpenAI provider pointed at a fake server, with the
// in-process web_search tool configured as the provider's injected web search.
type injectedToolsAccount struct{ baseURL string }

func (a *injectedToolsAccount) GetConfiguredProviders() ([]schemas.ModelProvider, error) {
	return []schemas.ModelProvider{schemas.OpenAI}, nil
}

func (a *injectedToolsAccount) GetKeysForProvider(ctx context.Context, provider schemas.ModelProvider) ([]schemas.Key, error) {
	return []schemas.Key{{ID: "k1", Value: *schemas.NewSecretVar("sk-test"), Models: schemas.WhiteList{"*"}, Weight: 1}}, nil
}

func (a *injectedToolsAccount) GetConfigForProvider(provider schemas.ModelProvider) (*schemas.ProviderConfig, error) {
	network := schemas.DefaultNetworkConfig
	network.BaseURL = a.baseURL
	network.MaxRetries = 0
	return &schemas.ProviderConfig{
		NetworkConfig:            network,
		ConcurrencyAndBufferSize: schemas.DefaultConcurrencyAndBufferSize,
		InjectedTools: &schemas.InjectedToolsConfig{
			WebSearch: &schemas.InjectedToolRef{MCPClientName: "bifrostInternal", ToolName: "web_search"},
		},
	}, nil
}

func setupInjectedToolsBifrost(t *testing.T, fake *fakeOpenAI, searches *[]string) *bifrost.Bifrost {
	t.Helper()
	server := httptest.NewServer(fake)
	t.Cleanup(server.Close)

	b, err := bifrost.Init(context.Background(), schemas.BifrostConfig{
		Account: &injectedToolsAccount{baseURL: server.URL},
		Logger:  bifrost.NewDefaultLogger(schemas.LogLevelError),
		MCPConfig: &schemas.MCPConfig{
			// Regular MCP auto-injection stays off: the provider config alone must put
			// the tool on the wire.
			ToolManagerConfig: &schemas.MCPToolManagerConfig{DisableAutoToolInject: true},
		},
	})
	require.NoError(t, err)
	t.Cleanup(b.Shutdown)

	var mu sync.Mutex
	schema := schemas.ChatTool{
		Type: schemas.ChatToolTypeFunction,
		Function: &schemas.ChatToolFunction{
			Name: "web_search",
			Parameters: &schemas.ToolFunctionParameters{
				Type:       "object",
				Properties: schemas.NewOrderedMapFromPairs(schemas.KV("query", map[string]any{"type": "string"})),
				Required:   []string{"query"},
			},
		},
	}
	require.NoError(t, b.RegisterMCPTool("web_search", "Search the web", func(args any) (string, error) {
		mu.Lock()
		defer mu.Unlock()
		query, _ := args.(map[string]any)["query"].(string)
		*searches = append(*searches, query)
		return "Paris: sunny, 24C", nil
	}, schema))
	return b
}

// TestInjectedWebSearch_ChatEndToEnd drives a chat request through Bifrost against a
// provider whose config injects an MCP web search tool. The client sends Anthropic's
// native web search tool and an include list that excludes the MCP tool; Bifrost must
// replace the native tool, run the search itself, and return only the final answer.
func TestInjectedWebSearch_ChatEndToEnd(t *testing.T) {
	fake := &fakeOpenAI{replies: []string{
		chatCompletionJSON("", "tool_calls", injectedSearchTool),
		chatCompletionJSON("It is sunny in Paris.", "stop", ""),
	}}
	var searches []string
	b := setupInjectedToolsBifrost(t, fake, &searches)

	ctx := createTestContext()
	ctx.SetValue(schemas.MCPContextKeyIncludeTools, []string{"other-tool"})
	resp, bifrostErr := b.ChatCompletionRequest(ctx, &schemas.BifrostChatRequest{
		Provider: schemas.OpenAI,
		Model:    "gpt-4o",
		Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("Weather in Paris?")}}},
		Params: &schemas.ChatParameters{
			Tools: []schemas.ChatTool{{Type: "web_search_20250305", Name: "web_search"}},
		},
	})

	require.Nil(t, bifrostErr, "%v", bifrostErr)
	assert.Equal(t, []string{"weather in paris"}, searches, "Bifrost ran the search itself")
	require.Len(t, fake.bodies, 2)
	assert.Equal(t, []string{injectedSearchTool}, fake.toolNames(0), "the native tool is replaced by the MCP tool")

	messages, _ := fake.bodies[1]["messages"].([]any)
	require.Len(t, messages, 3, "user, assistant tool call, tool result")
	toolResult := messages[2].(map[string]any)
	assert.Equal(t, "tool", toolResult["role"])
	assert.Contains(t, toolResult["content"], "sunny")

	msg := resp.Choices[0].Message
	assert.Equal(t, "It is sunny in Paris.", *msg.Content.ContentStr)
	assert.True(t, msg.ChatAssistantMessage == nil || len(msg.ToolCalls) == 0, "the client never sees the injected call")
	assert.Equal(t, 20, resp.Usage.PromptTokens, "usage covers both model turns")
}

// TestInjectedWebSearch_ClientToolCallsPassThrough pins that injection does not take
// over the client's own tools: they stay declared next to the injected one, and a call
// to them comes back to the client unexecuted.
func TestInjectedWebSearch_ClientToolCallsPassThrough(t *testing.T) {
	fake := &fakeOpenAI{replies: []string{
		chatCompletionJSON("", "tool_calls", "get_weather"),
	}}
	var searches []string
	b := setupInjectedToolsBifrost(t, fake, &searches)

	resp, bifrostErr := b.ChatCompletionRequest(createTestContext(), &schemas.BifrostChatRequest{
		Provider: schemas.OpenAI,
		Model:    "gpt-4o",
		Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("Weather?")}}},
		Params: &schemas.ChatParameters{Tools: []schemas.ChatTool{{
			Type: schemas.ChatToolTypeFunction, Function: &schemas.ChatToolFunction{Name: "get_weather"},
		}}},
	})

	require.Nil(t, bifrostErr, "%v", bifrostErr)
	assert.Empty(t, searches)
	assert.Equal(t, []string{"get_weather", injectedSearchTool}, fake.toolNames(0))
	require.Len(t, resp.Choices[0].Message.ToolCalls, 1, "a client tool call goes back to the client")
	assert.Equal(t, "get_weather", *resp.Choices[0].Message.ToolCalls[0].Function.Name)
}

#!/bin/bash
# Run all tests with proper error handling

echo "=========================================="
echo "🧪 RUNNING ALL ASYNCLLM TESTS"
echo "=========================================="

# Set colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Error Handling
echo -e "\n${YELLOW}1. Running Error Handling Tests...${NC}"
python tests/test_error_handling.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Error handling tests passed${NC}"
else
    echo -e "${RED}❌ Error handling tests failed${NC}"
fi

# Test 2: Concurrent Requests (optional - requires model)
echo -e "\n${YELLOW}2. Running Concurrent Request Tests...${NC}"
echo -e "${YELLOW}   (Skipping - requires model loaded)${NC}"
# python tests/test_concurrent_requests.py

# Test 3: Resource Cleanup
echo -e "\n${YELLOW}3. Running Resource Cleanup Tests...${NC}"
python tests/test_resource_cleanup.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Resource cleanup tests passed${NC}"
else
    echo -e "${RED}❌ Resource cleanup tests failed${NC}"
fi

echo -e "\n=========================================="
echo -e "🏁 TEST SUITE COMPLETE"
echo "=========================================="
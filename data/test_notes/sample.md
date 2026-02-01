# Authentication

Web authentication verifies user identity before granting access to protected resources.

## JWT Basics

JSON Web Tokens (JWT) are compact, self-contained tokens used for stateless authentication. A JWT has three parts: header, payload, and signature.

The header specifies the algorithm. The payload contains claims (user ID, expiration). The signature ensures the token hasn't been tampered with.

## OAuth 2.0

OAuth 2.0 is an authorization framework that allows third-party applications to access user resources without exposing credentials.

### Authorization Code Flow

The most secure OAuth flow for server-side apps:

1. User clicks "Login with Google"
2. Redirect to Google's auth server
3. User grants permission
4. Google redirects back with an authorization code
5. Your server exchanges the code for tokens

### Client Credentials Flow

Used for machine-to-machine communication where no user is involved. The client authenticates directly with the auth server using its own credentials.

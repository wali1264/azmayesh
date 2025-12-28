import React, { createContext, useContext, ReactNode, useState, useEffect, useCallback } from 'react';
import type {
    Product, ProductBatch, SaleInvoice, PurchaseInvoice, PurchaseInvoiceItem, InvoiceItem,
    Customer, Supplier, Employee, Expense, Service, StoreSettings, CartItem,
    CustomerTransaction, SupplierTransaction, PayrollTransaction, ActivityLog,
    User, Role, Permission, AppState
} from './types';
import { formatCurrency } from './utils/formatters';
import { api } from './services/supabaseService';
import { supabase } from './utils/supabaseClient';

interface AppContextType extends AppState {
    showToast: (message: string) => void;
    isLoading: boolean;
    
    // Auth
    login: (identifier: string, password: string, type: 'admin' | 'staff') => Promise<{ success: boolean; message: string; pending?: boolean; locked?: boolean }>;
    signup: (email: string, password: string) => Promise<{ success: boolean; message: string }>;
    logout: () => Promise<{ success: boolean; message: string }>;
    hasPermission: (permission: Permission) => boolean;
    
    // Users & Roles
    addUser: (user: Omit<User, 'id'>) => Promise<{ success: boolean; message: string }>;
    updateUser: (user: Partial<User> & { id: string }) => Promise<{ success: boolean; message: string }>;
    deleteUser: (userId: string) => Promise<void>;
    addRole: (role: Omit<Role, 'id'>) => Promise<{ success: boolean; message: string }>;
    updateRole: (role: Role) => Promise<{ success: boolean; message: string }>;
    deleteRole: (roleId: string) => Promise<void>;

    // Backup & Restore
    exportData: () => void;
    importData: (file: File) => void;

    // Inventory Actions
    addProduct: (product: Omit<Product, 'id' | 'batches'>, firstBatch: Omit<ProductBatch, 'id'>) => { success: boolean; message: string }; 
    updateProduct: (product: Product) => { success: boolean; message: string };
    deleteProduct: (productId: string) => void;
    
    // POS Actions
    addToCart: (itemToAdd: Product | Service, type: 'product' | 'service') => { success: boolean; message: string };
    updateCartItemQuantity: (itemId: string, itemType: 'product' | 'service', newQuantity: number) => { success: boolean; message: string };
    updateCartItemFinalPrice: (itemId: string, itemType: 'product' | 'service', finalPrice: number) => void;
    removeFromCart: (itemId: string, itemType: 'product' | 'service') => void;
    completeSale: (cashier: string, customerId?: string) => Promise<{ success: boolean; invoice?: SaleInvoice; message: string }>;
    beginEditSale: (invoiceId: string) => { success: boolean; message: string; customerId?: string; };
    cancelEditSale: () => void;
    addSaleReturn: (originalInvoiceId: string, returnItems: { id: string; type: 'product' | 'service'; quantity: number }[], cashier: string) => { success: boolean, message: string };
    setInvoiceTransientCustomer: (invoiceId: string, customerName: string) => Promise<void>;
    
    // Purchase Actions
    addPurchaseInvoice: (invoiceData: Omit<PurchaseInvoice, 'id' | 'totalAmount' | 'items' | 'type' | 'originalInvoiceId'> & { items: Omit<PurchaseInvoiceItem, 'productName'>[] }) => { success: boolean, message: string };
    beginEditPurchase: (invoiceId: string) => { success: boolean; message: string };
    cancelEditPurchase: () => void;
    updatePurchaseInvoice: (invoiceData: Omit<PurchaseInvoice, 'id' | 'totalAmount' | 'items' | 'type' | 'originalInvoiceId'> & { items: Omit<PurchaseInvoiceItem, 'productName'>[] }) => { success: boolean, message: string };
    addPurchaseReturn: (originalInvoiceId: string, returnItems: { productId: string; lotNumber: string, quantity: number }[]) => { success: boolean; message: string };

    // Settings
    updateSettings: (newSettings: StoreSettings) => void;
    
    // Services
    addService: (service: Omit<Service, 'id'>) => void;
    deleteService: (serviceId: string) => void;
    
    // Accounting
    addSupplier: (supplier: Omit<Supplier, 'id' | 'balance'>, initialBalance?: { amount: number, type: 'creditor' | 'debtor', currency: 'AFN' | 'USD', exchangeRate?: number }) => void;
    deleteSupplier: (id: string) => void;
    addSupplierPayment: (supplierId: string, amount: number, description: string, currency?: 'AFN' | 'USD', exchangeRate?: number) => SupplierTransaction;
    
    addCustomer: (customer: Omit<Customer, 'id' | 'balance'>, initialBalance?: { amount: number, type: 'creditor' | 'debtor', currency: 'AFN' | 'USD', exchangeRate?: number }) => void;
    deleteCustomer: (id: string) => void;
    addCustomerPayment: (customerId: string, amount: number, description: string) => CustomerTransaction;
    
    addEmployee: (employee: Omit<Employee, 'id'|'balance'>) => void;
    addEmployeeAdvance: (employeeId: string, amount: number) => void;
    processAndPaySalaries: () => { success: boolean; message: string };
    addExpense: (expense: Omit<Expense, 'id'>) => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

const getDeviceId = () => {
    let id = localStorage.getItem('ketabestan_device_id');
    if (!id) {
        id = crypto.randomUUID();
        localStorage.setItem('ketabestan_device_id', id);
    }
    return id;
};

const getDefaultState = (): AppState => {
    return {
        products: [], saleInvoices: [], purchaseInvoices: [], customers: [],
        suppliers: [], employees: [], expenses: [], services: [],
        storeSettings: {
            storeName: 'کتابستان', address: '', phone: '', lowStockThreshold: 10,
            expiryThresholdMonths: 3, currencyName: 'افغانی', currencySymbol: 'AFN'
        },
        cart: [], customerTransactions: [], supplierTransactions: [], payrollTransactions: [],
        activities: [], saleInvoiceCounter: 0, editingSaleInvoiceId: null, editingPurchaseInvoiceId: null,
        isAuthenticated: false, currentUser: null,
        users: [],
        roles: [],
    };
};

const generateNextId = (prefix: string, ids: string[]): string => {
    let max = 0;
    const regex = new RegExp(`^${prefix}(\\d+)$`); 
    for (const id of ids) {
        const match = id.match(regex);
        if (match) {
             const num = parseInt(match[1], 10);
             if (!isNaN(num)) {
                 if (num > max) max = num;
             }
        }
    }
    return `${prefix}${max + 1}`;
};

export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const [state, setState] = useState<AppState>(getDefaultState());
    const [isLoading, setIsLoading] = useState(true);
    const [toastMessage, setToastMessage] = useState('');

    const showToast = useCallback((message: string) => setToastMessage(message), []);

    const fetchData = useCallback(async () => {
        setIsLoading(true);
        try {
            const [settings, users, roles, products, services, entities, transactions, invoices, activity] = await Promise.all([
                api.getSettings().catch(() => ({})),
                api.getUsers().catch(() => []),
                api.getRoles().catch(() => []),
                api.getProducts().catch(() => []),
                api.getServices().catch(() => []),
                api.getEntities().catch(() => ({ customers: [], suppliers: [], employees: [], expenses: [] })),
                api.getTransactions().catch(() => ({ customerTransactions: [], supplierTransactions: [], payrollTransactions: [] })),
                api.getInvoices().catch(() => ({ saleInvoices: [], purchaseInvoices: [] })),
                api.getActivities().catch(() => [])
            ]);

            const { data: { session } } = await supabase.auth.getSession();
            let isAuth = false;
            let restoredUser = null;

            if (session?.user) {
                const profile = await api.getProfile(session.user.id);
                const deviceId = getDeviceId();
                
                if (profile && profile.is_approved) {
                    if (!profile.current_device_id || profile.current_device_id === deviceId) {
                        isAuth = true;
                        restoredUser = { id: session.user.id, username: session.user.email || 'Admin', roleId: 'admin-role' };
                        
                        // Critical check: Attempt to bind device if empty
                        if (!profile.current_device_id) {
                            const success = await api.updateProfile(session.user.id, { current_device_id: deviceId });
                            if (!success) {
                                isAuth = false; // Kick out if DB cannot record device
                                await logout();
                            }
                        }
                    }
                } else if (!navigator.onLine && localStorage.getItem('ketabestan_offline_auth') === 'true') {
                    isAuth = true;
                    restoredUser = { id: session.user.id, username: session.user.email || 'Admin', roleId: 'admin-role' };
                }
            } else {
                const localStaff = localStorage.getItem('ketabestan_staff_user');
                if (localStaff) {
                    try {
                        const parsedStaff = JSON.parse(localStaff) as User;
                        const dbUser = users.find(u => u.id === parsedStaff.id);
                        if (dbUser) {
                            isAuth = true;
                            restoredUser = dbUser;
                        } else {
                            localStorage.removeItem('ketabestan_staff_user');
                        }
                    } catch(e) { localStorage.removeItem('ketabestan_staff_user'); }
                }
            }

            setState(prev => ({
                ...prev,
                storeSettings: settings as StoreSettings || prev.storeSettings,
                users,
                roles: roles.length > 0 ? roles : [{ id: 'admin-role', name: 'Admin', permissions: ['page:dashboard', 'page:inventory', 'page:pos', 'page:purchases', 'page:accounting', 'page:reports', 'page:settings', 'inventory:add_product', 'inventory:edit_product', 'inventory:delete_product', 'pos:create_invoice', 'pos:edit_invoice', 'pos:apply_discount', 'pos:create_credit_sale', 'purchase:create_invoice', 'purchase:edit_invoice', 'accounting:manage_suppliers', 'accounting:manage_customers', 'accounting:manage_payroll', 'accounting:manage_expenses', 'settings:manage_store', 'settings:manage_users', 'settings:manage_backup', 'settings:manage_services', 'settings:manage_alerts'] }],
                products,
                services,
                customers: entities.customers,
                suppliers: entities.suppliers,
                employees: entities.employees,
                expenses: entities.expenses,
                customerTransactions: transactions.customerTransactions,
                supplierTransactions: transactions.supplierTransactions,
                payrollTransactions: transactions.payrollTransactions,
                saleInvoices: invoices.saleInvoices, 
                purchaseInvoices: invoices.purchaseInvoices,
                activities: activity,
                saleInvoiceCounter: invoices.saleInvoices.length,
                isAuthenticated: isAuth,
                currentUser: restoredUser
            }));
        } catch (error) {
            console.error("Error fetching data:", error);
            showToast("⚠️ خطا در دریافت اطلاعات ثانویه.");
        } finally {
            setIsLoading(false);
        }
    }, [showToast]);

    useEffect(() => {
        fetchData();
    }, [fetchData]);

    const login = async (identifier: string, password: string, type: 'admin' | 'staff'): Promise<{ success: boolean; message: string; pending?: boolean; locked?: boolean }> => {
        if (type === 'admin') {
            try {
                const { data, error } = await supabase.auth.signInWithPassword({ email: identifier, password });
                if (error) return { success: false, message: 'ایمیل یا رمز عبور اشتباه است.' };
                
                const profile = await api.getProfile(data.user.id);
                if (!profile) return { success: false, message: 'پروفایل یافت نشد.' };

                if (!profile.is_approved) return { success: false, message: 'حساب در انتظار تایید مدیریت است.', pending: true };

                const deviceId = getDeviceId();
                if (profile.current_device_id && profile.current_device_id !== deviceId) {
                    return { success: false, message: 'این حساب در دستگاه دیگری فعال است. ابتدا از آن خارج شوید.', locked: true };
                }

                // Strict Binding during Login
                if (!profile.current_device_id) {
                    const success = await api.updateProfile(data.user.id, { current_device_id: deviceId });
                    if (!success) {
                        await supabase.auth.signOut();
                        return { success: false, message: 'خطا در ثبت هویت دستگاه. لطفاً با ادمین دیتابیس تماس بگیرید.' };
                    }
                }

                localStorage.setItem('ketabestan_offline_auth', 'true');
                await fetchData();
                return { success: true, message: '✅ ورود موفق' };
            } catch (e) {
                return { success: false, message: '❌ خطا در اتصال به سرور.' };
            }
        } else {
            const user = await api.verifyStaffCredentials(identifier, password);
            if (user) {
                localStorage.setItem('ketabestan_staff_user', JSON.stringify(user));
                await fetchData();
                return { success: true, message: `✅ خوش آمدید ${user.username}` };
            } else {
                return { success: false, message: 'نام کاربری یا رمز عبور اشتباه است.' };
            }
        }
    };

    const signup = async (email: string, password: string): Promise<{ success: boolean; message: string }> => {
        try {
            const { error } = await supabase.auth.signUp({ email, password });
            if (error) return { success: false, message: error.message };
            return { success: true, message: '✅ ثبت‌نام انجام شد. لطفاً ایمیل خود را تایید کنید.' };
        } catch (e) {
            return { success: false, message: '❌ خطا در ثبت‌نام.' };
        }
    };

    const logout = async (): Promise<{ success: boolean; message: string }> => {
        const localStaff = localStorage.getItem('ketabestan_staff_user');
        if (localStaff) {
            localStorage.removeItem('ketabestan_staff_user');
            setState(prev => ({ ...prev, isAuthenticated: false, currentUser: null }));
            return { success: true, message: 'خروج موفق' };
        }

        if (!navigator.onLine) {
            showToast("⚠️ برای خروج باید به اینترنت متصل باشید.");
            return { success: false, message: 'عدم اتصال به اینترنت' };
        }

        try {
            const { data: { user } } = await supabase.auth.getUser();
            if (user) {
                await api.updateProfile(user.id, { current_device_id: null });
            }
            await supabase.auth.signOut();
            localStorage.removeItem('ketabestan_offline_auth');
            setState(prev => ({ ...prev, isAuthenticated: false, currentUser: null }));
            return { success: true, message: 'خروج موفق' };
        } catch (e) {
            return { success: false, message: 'خطا در خروج' };
        }
    };

    const hasPermission = (permission: Permission): boolean => {
        if (!state.currentUser) return false;
        const userRole = state.roles.find(r => r.id === state.currentUser!.roleId);
        return userRole?.permissions.includes(permission) ?? false;
    };
    
    // --- Activities ---
    const addActivityLocal = async (type: ActivityLog['type'], description: string, user: string, refId?: string, refType?: ActivityLog['refType']) => {
        const newActivity: ActivityLog = {
            id: crypto.randomUUID(), type, description, timestamp: new Date().toISOString(), user, refId, refType
        };
        setState(prev => ({ ...prev, activities: [newActivity, ...prev.activities] }));
        try {
            await api.addActivity(newActivity);
        } catch (e) { console.error("Failed to log activity", e); }
        return newActivity;
    };

    // --- Users & Roles ---
    const addUser = async (userData: Omit<User, 'id'>) => {
        try {
            const newUser = await api.addUser(userData);
            setState(prev => ({ ...prev, users: [...prev.users, newUser] }));
            return { success: true, message: '✅ کاربر جدید افزوده شد.' };
        } catch (e) { return { success: false, message: '❌ خطا.' }; }
    };

    const updateUser = async (userData: Partial<User> & { id: string }) => {
        try {
             await api.updateUser(userData);
             setState(prev => ({ ...prev, users: prev.users.map(u => u.id === userData.id ? { ...u, ...userData } : u) }));
             return { success: true, message: '✅ بروزرسانی شد.' };
        } catch (e) { return { success: false, message: '❌ خطا.' }; }
    };

    const deleteUser = async (userId: string) => {
         try {
            await api.deleteUser(userId);
            setState(prev => ({ ...prev, users: prev.users.filter(u => u.id !== userId) }));
            showToast("✅ کاربر حذف شد.");
         } catch (e) { showToast("❌ خطا."); }
    };

    const addRole = async (roleData: Omit<Role, 'id'>) => {
        try {
            const newRole = await api.addRole(roleData);
            setState(prev => ({ ...prev, roles: [...prev.roles, newRole] }));
            return { success: true, message: '✅ نقش جدید افزوده شد.' };
        } catch (e) { return { success: false, message: '❌ خطا.' }; }
    };

    const updateRole = async (roleData: Role) => {
        try {
            await api.updateRole(roleData);
            setState(prev => ({ ...prev, roles: prev.roles.map(r => r.id === roleData.id ? roleData : r) }));
            return { success: true, message: '✅ نقش بروزرسانی شد.' };
        } catch (e) { return { success: false, message: '❌ خطا.' }; }
    };

    const deleteRole = async (roleId: string) => {
        try {
            await api.deleteRole(roleId);
            setState(prev => ({ ...prev, roles: prev.roles.filter(r => r.id !== roleId) }));
        } catch(e) { showToast("❌ خطا."); }
    };

    // --- POS ---
    const addToCart = (itemToAdd: Product | Service, type: 'product' | 'service') => {
        let success = false, message = '';
        setState(prev => {
            const existingItemIndex = prev.cart.findIndex(item => item.id === itemToAdd.id && item.type === type);
            const totalStock = type === 'product' ? (itemToAdd as Product).batches.reduce((sum, b) => sum + b.stock, 0) : Infinity;
            if (existingItemIndex > -1) {
                const updatedCart = [...prev.cart];
                const existingItem = updatedCart[existingItemIndex];
                if (type === 'product' && existingItem.quantity >= totalStock) {
                    message = `حداکثر موجودی برای "${existingItem.name}" در سبد خرید است.`; return prev; 
                }
                updatedCart[existingItemIndex] = { ...existingItem, quantity: existingItem.quantity + 1 };
                success = true;
                return { ...prev, cart: updatedCart };
            } else {
                if (type === 'product' && totalStock < 1) {
                    message = `موجودی محصول "${itemToAdd.name}" تمام شده است.`; return prev; 
                }
                success = true;
                return { ...prev, cart: [...prev.cart, { ...itemToAdd, quantity: 1, type } as any] };
            }
        });
        return { success, message };
    };

    const updateCartItemQuantity = (itemId: string, itemType: 'product' | 'service', newQuantity: number) => {
        if (newQuantity < 0) return { success: false, message: 'مقدار نامعتبر' };
        let success = true, message = '';
        setState(prev => {
            const cart = [...prev.cart];
            const itemIndex = cart.findIndex(i => i.id === itemId && i.type === itemType);
            if (itemIndex === -1) return prev;
            if (itemType === 'product') {
                 const productInStock = prev.products.find(p => p.id === itemId);
                 const totalStock = productInStock?.batches.reduce((sum, b) => sum + b.stock, 0) || 0;
                 if (newQuantity > totalStock) {
                    message = `موجودی محصول فقط ${totalStock} عدد است.`;
                    cart[itemIndex] = { ...cart[itemIndex], quantity: totalStock };
                    return { ...prev, cart };
                 }
            }
            cart[itemIndex] = { ...cart[itemIndex], quantity: newQuantity };
            if (newQuantity === 0) return { ...prev, cart: cart.filter(i => !(i.id === itemId && i.type === itemType)) };
            return { ...prev, cart };
        });
        return { success, message };
    };

    const updateCartItemFinalPrice = (itemId: string, itemType: 'product' | 'service', finalPrice: number) => {
        setState(prev => ({
            ...prev, cart: prev.cart.map(item =>
                (item.id === itemId && item.type === itemType && item.type === 'product')
                    ? { ...item, finalPrice: finalPrice } 
                    : item
            )
        }));
    };
    
    const removeFromCart = (itemId: string, itemType: 'product' | 'service') => {
        setState(prev => ({ ...prev, cart: prev.cart.filter(item => !(item.id === itemId && item.type === itemType)) }));
    };

    const completeSale = async (cashier: string, customerId?: string): Promise<{ success: boolean; invoice?: SaleInvoice; message: string }> => {
        const { cart, products, storeSettings, editingSaleInvoiceId, customers, saleInvoices } = state;
        if (cart.length === 0) return { success: false, message: "سبد خرید خالی است!" };
        
        const subtotal = cart.reduce((total, item) => ((item.type === 'product' ? item.salePrice : item.price) * item.quantity) + total, 0);
        const totalAmount = cart.reduce((total, item) => {
            const price = (item.type === 'product' && item.finalPrice !== undefined) ? item.finalPrice : (item.type === 'product' ? item.salePrice : item.price);
            return (price * item.quantity) + total;
        }, 0);

        const stockUpdates: {batchId: string, newStock: number}[] = [];
        const saleItemsWithPurchasePrice: CartItem[] = [];
        const updatedProducts = JSON.parse(JSON.stringify(products));

        for (const item of cart) {
            if (item.type === 'service') { saleItemsWithPurchasePrice.push(item); continue; }
            const p = updatedProducts.find((p: Product) => p.id === item.id);
            let quantityToDeduct = item.quantity;
            let totalPurchaseValue = 0;
            p.batches.sort((a: any, b: any) => new Date(a.expiryDate || a.purchaseDate).getTime() - new Date(b.expiryDate || b.purchaseDate).getTime());
            for (const batch of p.batches) {
                if (quantityToDeduct <= 0) break;
                const deduct = Math.min(quantityToDeduct, batch.stock);
                batch.stock -= deduct;
                quantityToDeduct -= deduct;
                totalPurchaseValue += deduct * batch.purchasePrice;
                stockUpdates.push({ batchId: batch.id, newStock: batch.stock });
            }
            saleItemsWithPurchasePrice.push({ ...item, purchasePrice: totalPurchaseValue / item.quantity });
        }

        const invoiceId = editingSaleInvoiceId || generateNextId('F', saleInvoices.map(i => i.id));
        const finalInvoice: SaleInvoice = { id: invoiceId, type: 'sale', items: saleItemsWithPurchasePrice, subtotal, totalAmount, totalDiscount: subtotal - totalAmount, timestamp: new Date().toISOString(), cashier, customerId };

        let customerUpdate;
        if (customerId) {
            const customer = customers.find(c => c.id === customerId)!;
            customerUpdate = { id: customerId, newBalance: customer.balance + totalAmount, transaction: { id: crypto.randomUUID(), customerId, type: 'credit_sale' as const, amount: totalAmount, date: new Date().toISOString(), description: `فاکتور فروش #${invoiceId}`, invoiceId } };
        }

        try {
            if (editingSaleInvoiceId) {
                await api.updateSale(invoiceId, finalInvoice, [], [], undefined);
                showToast("✅ فاکتور ویرایش شد.");
            } else {
                await api.createSale(finalInvoice, stockUpdates, customerUpdate);
                showToast("✅ فاکتور ثبت شد.");
            }
            await fetchData();
            setState(prev => ({ ...prev, cart: [], editingSaleInvoiceId: null }));
            return { success: true, invoice: finalInvoice, message: 'فاکتور ثبت شد.' };
        } catch (e) { return { success: false, message: 'خطا در ثبت فاکتور.' }; }
    };

    const beginEditSale = (invoiceId: string) => {
        const invoice = state.saleInvoices.find(i => i.id === invoiceId);
        if (!invoice) return { success: false, message: "فاکتور یافت نشد." };
        setState(prev => ({ ...prev, editingSaleInvoiceId: invoiceId, cart: invoice.items.map(i => ({ ...i } as CartItem)) }));
        return { success: true, message: "آماده ویرایش.", customerId: invoice.customerId };
    };

    const cancelEditSale = () => setState(prev => ({ ...prev, editingSaleInvoiceId: null, cart: [] }));

    const addSaleReturn = (originalInvoiceId: string, returnItems: { id: string; type: 'product' | 'service'; quantity: number }[], cashier: string) => {
        const originalInvoice = state.saleInvoices.find(i => i.id === originalInvoiceId);
        if (!originalInvoice) return { success: false, message: "فاکتور اصلی یافت نشد." };
        
        let returnTotal = 0;
        const detailedReturnItems = returnItems.map(ri => {
            const originalItem = originalInvoice.items.find(i => i.id === ri.id && i.type === ri.type);
            if (!originalItem) return null;
            const price = (originalItem.type === 'product' && originalItem.finalPrice !== undefined) ? originalItem.finalPrice : (originalItem.type === 'product' ? originalItem.salePrice : originalItem.price);
            returnTotal += price * ri.quantity;
            return { ...originalItem, quantity: ri.quantity };
        }).filter(Boolean) as CartItem[];

        const returnInvoice: SaleInvoice = {
            id: generateNextId('R', state.saleInvoices.map(i => i.id)),
            type: 'return',
            originalInvoiceId,
            items: detailedReturnItems,
            subtotal: returnTotal,
            totalAmount: returnTotal,
            totalDiscount: 0,
            timestamp: new Date().toISOString(),
            cashier,
            customerId: originalInvoice.customerId
        };

        api.createSaleReturn(returnInvoice, detailedReturnItems.filter(i => i.type === 'product').map(i => ({ productId: i.id, quantity: i.quantity })), originalInvoice.customerId ? { id: originalInvoice.customerId, amount: returnTotal } : undefined)
           .then(() => { fetchData(); showToast("✅ مرجوعی ثبت شد."); });
        return { success: true, message: "در حال ثبت مرجوعی..." };
    };

    const setInvoiceTransientCustomer = async (invoiceId: string, customerName: string) => {
         await api.updateSaleInvoiceMetadata(invoiceId, { original_invoice_id: customerName });
         setState(prev => ({ ...prev, saleInvoices: prev.saleInvoices.map(inv => inv.id === invoiceId ? { ...inv, originalInvoiceId: customerName } : inv) }));
    };

    // --- Inventory / Purchases ---
    const addProduct = (productData: Omit<Product, 'id' | 'batches'>, firstBatchData: Omit<ProductBatch, 'id'>) => {
        api.addProduct(productData, firstBatchData).then(newProduct => {
             addActivityLocal('inventory', `محصول جدید "${productData.name}" اضافه شد`, state.currentUser!.username, newProduct.id, 'product');
             setState(prev => ({ ...prev, products: [...prev.products, newProduct] }));
             showToast('✅ محصول ذخیره شد.');
        });
        return { success: true, message: 'در حال ذخیره...' };
    };
    
    const updateProduct = (productData: Product) => {
         api.updateProduct(productData).then(() => {
             setState(prev => ({ ...prev, products: prev.products.map(p => p.id === productData.id ? productData : p) }));
             showToast('✅ محصول ویرایش شد.');
         });
        return { success: true, message: 'در حال ویرایش...' };
    };
    
    const deleteProduct = (productId: string) => {
        api.deleteProduct(productId).then(() => {
            setState(prev => ({ ...prev, products: prev.products.filter(p => p.id !== productId) }));
            showToast('✅ محصول حذف شد.');
        });
    };

    const addPurchaseInvoice = (invoiceData: any) => {
        const { products, purchaseInvoices, suppliers } = state;
        const invoiceId = generateNextId('P', purchaseInvoices.map(i => i.id));
        
        let totalAmount = 0;
        const newBatches: any[] = [];
        const itemsWithNames: PurchaseInvoiceItem[] = [];

        for (const item of invoiceData.items) {
            const product = products.find(p => p.id === item.productId);
            const batchId = crypto.randomUUID();
            const batch: any = {
                id: batchId,
                productId: item.productId,
                lotNumber: item.lotNumber,
                stock: item.quantity,
                purchasePrice: item.purchasePrice,
                purchaseDate: invoiceData.timestamp,
                expiryDate: item.expiryDate
            };
            newBatches.push(batch);
            totalAmount += item.quantity * item.purchasePrice;
            itemsWithNames.push({
                ...item,
                productName: product?.name || 'ناشناس'
            });
        }

        if (invoiceData.currency === 'USD') {
            totalAmount = Math.round(totalAmount * (invoiceData.exchangeRate || 1));
        }

        const finalInvoice: PurchaseInvoice = {
            id: invoiceId,
            type: 'purchase',
            supplierId: invoiceData.supplierId,
            invoiceNumber: invoiceData.invoiceNumber,
            items: itemsWithNames,
            totalAmount: totalAmount,
            timestamp: invoiceData.timestamp,
            currency: invoiceData.currency,
            exchangeRate: invoiceData.exchangeRate
        };

        const supplier = suppliers.find(s => s.id === invoiceData.supplierId);
        const supplierUpdate = {
            id: invoiceData.supplierId,
            newBalance: (supplier?.balance || 0) + totalAmount,
            transaction: {
                id: crypto.randomUUID(),
                supplierId: invoiceData.supplierId,
                type: 'purchase' as const,
                amount: totalAmount,
                date: invoiceData.timestamp,
                description: `فاکتور خرید #${invoiceData.invoiceNumber || invoiceId}`,
                invoiceId
            }
        };

        api.createPurchase(finalInvoice, supplierUpdate, newBatches).then(() => { fetchData(); showToast("✅ فاکتور خرید ثبت شد."); });
        return { success: true, message: "در حال ثبت..." };
    };

    const beginEditPurchase = (invoiceId: string) => { setState(prev => ({ ...prev, editingPurchaseInvoiceId: invoiceId })); return { success: true, message: "ویرایش خرید." }; };
    const cancelEditPurchase = () => setState(prev => ({ ...prev, editingPurchaseInvoiceId: null }));

    const updatePurchaseInvoice = (invoiceData: any) => {
        const { editingPurchaseInvoiceId, purchaseInvoices, products } = state;
        if (!editingPurchaseInvoiceId) return { success: false, message: "فاکتور انتخاب نشده." };

        const originalInvoice = purchaseInvoices.find(i => i.id === editingPurchaseInvoiceId);
        if (!originalInvoice) return { success: false, message: "فاکتور یافت نشد." };

        let totalAmount = 0;
        const itemsWithNames: PurchaseInvoiceItem[] = [];
        for (const item of invoiceData.items) {
            const product = products.find(p => p.id === item.productId);
            totalAmount += Number(item.quantity) * Number(item.purchasePrice);
            itemsWithNames.push({ ...item, productName: product?.name || 'ناشناس' });
        }
        if (invoiceData.currency === 'USD') {
            totalAmount = Math.round(totalAmount * (invoiceData.exchangeRate || 1));
        }

        const updatedInvoice: PurchaseInvoice = {
            id: editingPurchaseInvoiceId,
            type: 'purchase',
            supplierId: invoiceData.supplierId,
            invoiceNumber: invoiceData.invoiceNumber,
            items: itemsWithNames,
            totalAmount,
            timestamp: invoiceData.timestamp,
            currency: invoiceData.currency,
            exchangeRate: invoiceData.exchangeRate
        };

        const supplierUpdate = originalInvoice.supplierId === invoiceData.supplierId 
            ? { id: invoiceData.supplierId, oldAmount: originalInvoice.totalAmount, newAmount: totalAmount }
            : undefined;

        api.updatePurchase(editingPurchaseInvoiceId, updatedInvoice, supplierUpdate).then(() => { fetchData(); showToast("✅ بروزرسانی شد."); });
        return { success: true, message: "در حال بروزرسانی..." };
    };

    const addPurchaseReturn = (originalInvoiceId: string, returnItems: any[]) => {
        const { purchaseInvoices } = state;
        const originalInvoice = purchaseInvoices.find(i => i.id === originalInvoiceId);
        if (!originalInvoice) return { success: false, message: "فاکتور اصلی یافت نشد." };

        let returnTotal = 0;
        const detailedReturnItems = returnItems.map(ri => {
            const originalItem = originalInvoice.items.find(i => i.productId === ri.productId && i.lotNumber === ri.lotNumber);
            if (!originalItem) return null;
            returnTotal += originalItem.purchasePrice * ri.quantity;
            return { ...originalItem, quantity: ri.quantity };
        }).filter(Boolean) as PurchaseInvoiceItem[];

        if (originalInvoice.currency === 'USD') {
            returnTotal = Math.round(returnTotal * (originalInvoice.exchangeRate || 1));
        }

        const returnInvoice: PurchaseInvoice = {
            id: generateNextId('PR', purchaseInvoices.map(i => i.id)),
            type: 'return',
            originalInvoiceId,
            supplierId: originalInvoice.supplierId,
            invoiceNumber: `R-${originalInvoice.invoiceNumber || originalInvoiceId}`,
            items: detailedReturnItems,
            totalAmount: returnTotal,
            timestamp: new Date().toISOString()
        };

        const stockDeductions = detailedReturnItems.map(i => ({ productId: i.productId, quantity: i.quantity, lotNumber: i.lotNumber }));
        
        api.createPurchaseReturn(returnInvoice, stockDeductions, { id: originalInvoice.supplierId, amount: returnTotal })
           .then(() => { fetchData(); showToast("✅ مرجوعی خرید ثبت شد."); });
        return { success: true, message: "در حال ثبت..." };
    };

    // --- Other Entities ---
    const updateSettings = (newSettings: StoreSettings) => api.updateSettings(newSettings).then(() => { setState(prev => ({ ...prev, storeSettings: newSettings })); showToast("✅ تنظیمات ذخیره شد."); });
    const addService = (service: any) => api.addService(service).then(() => fetchData());
    const deleteService = (id: string) => api.deleteService(id).then(() => fetchData());
    const addSupplier = (s: any, initialBalance?: any) => api.addSupplier(s).then(newS => { if (initialBalance) api.processPayment('supplier', newS.id, initialBalance.amount, { id: crypto.randomUUID(), supplierId: newS.id, type: 'purchase', amount: initialBalance.amount, date: new Date().toISOString(), description: 'تراز اول' }).then(() => fetchData()); else fetchData(); });
    const deleteSupplier = (id: string) => api.deleteSupplier(id).then(() => fetchData());
    const addCustomer = (c: any, initialBalance?: any) => api.addCustomer(c).then(newC => { if (initialBalance) api.processPayment('customer', newC.id, initialBalance.amount, { id: crypto.randomUUID(), customerId: newC.id, type: 'credit_sale', amount: initialBalance.amount, date: new Date().toISOString(), description: 'تراز اول' }).then(() => fetchData()); else fetchData(); });
    // FIX: Use api.deleteCustomer instead of direct db access which was causing import errors.
    const deleteCustomer = (id: string) => api.deleteCustomer(id).then(() => fetchData());
    const addEmployee = (e: any) => api.addEmployee(e).then(() => fetchData());
    const addExpense = (e: any) => api.addExpense(e).then(() => fetchData());
    const addSupplierPayment = (supplierId: string, amount: number, description: string, currency: any, rate: any) => {
        const supplier = state.suppliers.find(s => s.id === supplierId)!;
        const tx = { id: crypto.randomUUID(), supplierId, type: 'payment' as const, amount, date: new Date().toISOString(), description, currency };
        api.processPayment('supplier', supplierId, supplier.balance - (currency === 'USD' ? amount * rate : amount), tx).then(() => fetchData());
        return tx;
    };
    const addCustomerPayment = (customerId: string, amount: number, description: string) => {
        const customer = state.customers.find(c => c.id === customerId)!;
        const tx = { id: crypto.randomUUID(), customerId, type: 'payment' as const, amount, date: new Date().toISOString(), description };
        api.processPayment('customer', customerId, customer.balance - amount, tx).then(() => fetchData());
        return tx;
    };
    const addEmployeeAdvance = (employeeId: string, amount: number) => {
        const employee = state.employees.find(e => e.id === employeeId)!;
        api.processPayment('employee', employeeId, employee.balance + amount, { id: crypto.randomUUID(), employeeId, type: 'advance' as const, amount, date: new Date().toISOString(), description: 'مساعده' }).then(() => fetchData());
    };
    const processAndPaySalaries = () => {
        const { employees } = state;
        let total = 0;
        const txs: PayrollTransaction[] = [];
        employees.forEach(emp => {
            const net = emp.monthlySalary - emp.balance;
            if (net > 0) {
                txs.push({ id: crypto.randomUUID(), employeeId: emp.id, type: 'salary_payment', amount: net, date: new Date().toISOString(), description: 'حقوق' });
                total += net;
            }
        });
        if (total === 0) return { success: false, message: 'موردی نیست.' };
        api.processPayroll(employees.map(e => ({ id: e.id, balance: 0 as const })), txs, { id: crypto.randomUUID(), category: 'salary', description: 'حقوق', amount: total, date: new Date().toISOString() }).then(() => fetchData());
        return { success: true, message: 'در حال پردازش...' };
    };

    // --- Backup ---
    const exportData = () => {
        const dataStr = JSON.stringify(state, null, 2);
        const blob = new Blob([dataStr], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `Ketabestan_Backup_${new Date().toISOString().split('T')[0]}.json`;
        link.click();
        URL.revokeObjectURL(url);
    };
    const importData = async (file: File) => {
        const reader = new FileReader();
        reader.onload = async (e) => {
            try {
                const data = JSON.parse(e.target?.result as string) as AppState;
                await api.clearAndRestoreData(data);
                await fetchData();
                showToast("✅ بازیابی موفق.");
            } catch (err) { showToast("❌ خطا در فایل."); }
        };
        reader.readAsText(file);
    };

    if (isLoading) return <div className="flex items-center justify-center h-screen text-xl font-bold text-blue-600">در حال دریافت اطلاعات...</div>;

    return <AppContext.Provider value={{
        ...state, showToast, isLoading, login, signup, logout, hasPermission, addUser, updateUser, deleteUser, addRole, updateRole, deleteRole, exportData, importData,
        addProduct, updateProduct, deleteProduct, addToCart, updateCartItemQuantity, updateCartItemFinalPrice, removeFromCart, completeSale,
        beginEditSale, cancelEditSale, addSaleReturn, addPurchaseInvoice, beginEditPurchase, cancelEditPurchase, updatePurchaseInvoice, addPurchaseReturn,
        updateSettings, addService, deleteService, addSupplier, deleteSupplier, addSupplierPayment, addCustomer, deleteCustomer, addCustomerPayment,
        addEmployee, addEmployeeAdvance, processAndPaySalaries, addExpense, setInvoiceTransientCustomer
    }}>{children}</AppContext.Provider>;
};

export const useAppContext = (): AppContextType => {
    const context = useContext(AppContext);
    if (context === undefined) throw new Error('useAppContext must be used within an AppProvider');
    return context;
};